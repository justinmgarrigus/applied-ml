"""Sampled occupation rows for learned angular-sector wavefunctions.

This module never loads a many-body basis, a Hamiltonian matrix, or an ED
state. ``SphereOperator.sample_uniform`` uses integer completion counts to draw
roots uniformly with replacement in a fixed Ne,M sector. Exact connections are
built only for roots actually drawn. Cache growth may eventually visit every
configuration of a small sector; that is distinct from enumerating its basis.

The returned amplitude union is a *data batch*, not a selected Hilbert-space
basis: it changes each call, and contains all neighbors needed by the selected
roots. NN derivatives must remain active for every union amplitude.

For a complex amplitude vector ``a`` evaluated at ``batch['occupations']``::

    roots = a[batch['root_indices']]
    H_a = (batch['H_coefficients'] * a[batch['H_indices']]).sum(axis=1)
    K_a = (batch['K_coefficients'] * a[batch['K_indices']]).sum(axis=1)

Here K=L^2 is exact for every root. Optional importance sampling of H connections
is unbiased for H_a and its NN derivative, conditional on the sampled root.
It does not remove finite-batch bias of a subsequently formed ratio estimator.
Never square a noisy operator action and call it an unbiased squared residual.
"""
from __future__ import annotations

from collections import OrderedDict
from numbers import Integral
import time
from typing import Any

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def occupation_bits(occupation) -> int:
    """Python integer encoding; no machine-word orbital limit."""
    return sum(1 << int(index) for index in np.flatnonzero(occupation))


def occupations_from_bits(bits, norb):
    return np.asarray([[(int(value) >> orbital) & 1 for orbital in range(norb)]
                       for value in bits], dtype=np.int8)


def l2_connections(occupation, *, ne: int, flux: int):
    """Combined exact row of L_-L_+ + M(M+1) in canonical Slater order.

    An adjacent allowed hop has positive fermionic sign. At most Ne^2 ladder
    paths are generated, then duplicate final configurations are summed. The
    returned diagonal is first, even when its value is zero.
    """
    root=np.asarray(occupation)
    if root.shape != (flux+1,) or not np.all((root==0)|(root==1)):
        raise ValueError('occupation must be binary with flux+1 entries')
    occupied=np.flatnonzero(root)
    if len(occupied)!=ne:raise ValueError('occupation has the wrong particle count')
    initial=occupation_bits(root)
    m=float(occupied.sum()-ne*flux/2.)
    entries={initial:m*(m+1.)}
    for a_np in occupied:
        a=int(a_np)
        if a==flux or root[a+1]:continue
        intermediate=(initial ^ (1<<a)) | (1<<(a+1))
        up=np.sqrt(float((flux-a)*(a+1)))
        for b in range(1,flux+1):
            if not (intermediate & (1<<b)) or intermediate & (1<<(b-1)):continue
            destination=(intermediate ^ (1<<b)) | (1<<(b-1))
            coefficient=up*np.sqrt(float(b*(flux-b+1)))
            entries[destination]=entries.get(destination,0.)+coefficient
    ordered=[initial]+sorted(key for key,value in entries.items() if key!=initial and value!=0.)
    return occupations_from_bits(ordered,flux+1),np.asarray([entries[key] for key in ordered],dtype=np.float64)


class RootConnectionCache:
    """LRU connections for sampled roots only, bound to one operator object.

    ``max_rows=None`` keeps all visited roots. Bounded caches evict stored rows,
    but cumulative coverage sets remain counts of actual roots/NN inputs seen.
    No reference basis or configuration list is constructed.
    """
    def __init__(self, operator, max_rows: int | None=None):
        self.operator=operator
        self.max_rows=None if max_rows is None else _positive_integer(max_rows,'max_rows')
        self._rows=OrderedDict()
        self.visited_roots=set()
        self.visited_amplitude_configurations=set()
        self.hits=0;self.misses=0

    def connections(self, root):
        key=occupation_bits(root)
        self.visited_roots.add(key)
        if key in self._rows:
            self.hits+=1;self._rows.move_to_end(key)
            return self._rows[key]
        self.misses+=1
        h_neighbors,h_coeff=self.operator.connections(root)
        k_neighbors,k_coeff=l2_connections(root,ne=self.operator.ne,flux=self.operator.flux)
        result=(tuple(occupation_bits(n) for n in h_neighbors),np.asarray(h_coeff,dtype=np.float64),
                tuple(occupation_bits(n) for n in k_neighbors),np.asarray(k_coeff,dtype=np.float64))
        self._rows[key]=result
        if self.max_rows is not None and len(self._rows)>self.max_rows:self._rows.popitem(last=False)
        return result

    def statistics(self):
        return dict(cached_root_rows=len(self._rows),cumulative_unique_sampled_roots=len(self.visited_roots),
                    cumulative_unique_amplitude_configurations=len(self.visited_amplitude_configurations),
                    cache_hits=self.hits,cache_misses=self.misses,max_cached_root_rows=self.max_rows)


def _sample_h_connections(bits,coefficients,rng,count):
    absolute=np.abs(coefficients);normalization=float(absolute.sum())
    if normalization==0.:return (bits[0],),np.asarray([0.],dtype=np.float64)
    probabilities=absolute/normalization
    chosen=rng.choice(len(bits),size=count,replace=True,p=probabilities)
    combined={}
    for index in chosen:
        key=bits[int(index)]
        combined[key]=combined.get(key,0.)+float(coefficients[index]/(count*probabilities[index]))
    selected=tuple(sorted(combined))
    return selected,np.asarray([combined[key] for key in selected],dtype=np.float64)


def _padded_rows(rows,union_index,capacity,name):
    required=max(map(lambda row:len(row[0]),rows))
    width=required if capacity is None else _positive_integer(capacity,name)
    if width<required:raise ValueError(f'{name}={width} is smaller than the required {required}; no truncation is allowed')
    indices=np.zeros((len(rows),width),dtype=np.int32)
    coefficients=np.zeros((len(rows),width),dtype=np.float64)
    valid=np.zeros((len(rows),width),dtype=bool)
    for row,(bits,values) in enumerate(rows):
        size=len(bits);indices[row,:size]=[union_index[key] for key in bits]
        coefficients[row,:size]=values;valid[row,:size]=True
    return indices,coefficients,valid


def sample_operator_batch(operator,rng: np.random.Generator,batch_size: int,
                          cache: RootConnectionCache | None=None,*,
                          h_samples_per_root: int | None=None,
                          union_pad_to: int | None=None,
                          h_pad_to: int | None=None,
                          k_pad_to: int | None=None) -> dict[str,Any]:
    """Uniform roots plus a deduplicated exact-K/optional-sampled-H NN union.

    Optional capacities give fixed accelerator array shapes. Padding repeats
    the first valid configuration and has zero operator weights. Insufficient
    capacities raise before return; no connection or configuration is dropped.
    The ``occupation_mask`` identifies distinct, nonpadding NN input rows.

    ``h_samples_per_root=None`` means exact H. Otherwise each H row uses that
    many independent draws with replacement with p_j proportional to |H_nj|,
    returning the combined weights H_nj/(s*p_j). K remains exact, including its
    diagonal and all Ne^2 ladder paths. Uniform-root draws precede H draws.
    """
    started=time.perf_counter();batch_size=_positive_integer(batch_size,'batch_size')
    if not isinstance(rng,np.random.Generator):raise TypeError('rng must be a NumPy Generator')
    if cache is None:cache=RootConnectionCache(operator)
    if cache.operator is not operator:raise ValueError('The cache belongs to a different operator object')
    if h_samples_per_root is not None:h_samples_per_root=_positive_integer(h_samples_per_root,'h_samples_per_root')
    roots=operator.sample_uniform(rng,batch_size)
    root_bits=[occupation_bits(n) for n in roots]
    h_rows=[];k_rows=[];exact_h_counts=[];exact_k_counts=[]
    previous_hits=cache.hits;previous_misses=cache.misses
    for root in roots:
        h_bits,h_coeff,k_bits,k_coeff=cache.connections(root)
        exact_h_counts.append(len(h_bits));exact_k_counts.append(len(k_bits))
        if h_samples_per_root is not None:h_bits,h_coeff=_sample_h_connections(h_bits,h_coeff,rng,h_samples_per_root)
        h_rows.append((h_bits,h_coeff));k_rows.append((k_bits,k_coeff))
    union_bits=list(dict.fromkeys(root_bits))
    union_index={key:index for index,key in enumerate(union_bits)}
    for bits,_ in h_rows+k_rows:
        for key in bits:
            if key not in union_index:union_index[key]=len(union_bits);union_bits.append(key)
    unique_count=len(union_bits)
    capacity=unique_count if union_pad_to is None else _positive_integer(union_pad_to,'union_pad_to')
    if capacity<unique_count:raise ValueError(f'union_pad_to={capacity} is smaller than the required {unique_count}; no truncation is allowed')
    cache.visited_amplitude_configurations.update(union_bits)
    padded_bits=union_bits+[union_bits[0]]*(capacity-unique_count)
    h_indices,h_coeff,h_valid=_padded_rows(h_rows,union_index,h_pad_to,'h_pad_to')
    k_indices,k_coeff,k_valid=_padded_rows(k_rows,union_index,k_pad_to,'k_pad_to')
    mask=np.arange(capacity)<unique_count
    coverage=dict(root_rows=batch_size,unique_root_configurations=len(set(root_bits)),
        unique_H_neighbor_configurations=len(set(key for bits,_ in h_rows for key in bits)),
        unique_K_neighbor_configurations=len(set(key for bits,_ in k_rows for key in bits)),
        unique_amplitude_configurations=unique_count,NN_input_rows_including_padding=capacity,
        padding_rows=capacity-unique_count,exact_H_connections_per_root=exact_h_counts,
        exact_K_connections_per_root=exact_k_counts,returned_H_connections_per_root=[len(row[0]) for row in h_rows],
        returned_K_connections_per_root=[len(row[0]) for row in k_rows],
        sector_dimension_from_integer_DP=int(operator.dimension()),
        minibatch_unique_fraction_of_sector=unique_count/operator.dimension(),
        cache_hits_this_batch=cache.hits-previous_hits,cache_misses_this_batch=cache.misses-previous_misses,
        **cache.statistics())
    return dict(roots=np.asarray(roots,dtype=np.int8),occupations=occupations_from_bits(padded_bits,operator.norb),
        occupation_mask=mask,root_indices=np.asarray([union_index[key] for key in root_bits],dtype=np.int32),
        H_indices=h_indices,H_coefficients=h_coeff,H_mask=h_valid,
        K_indices=k_indices,K_coefficients=k_coeff,K_mask=k_valid,
        coverage=coverage,preprocessing_seconds=time.perf_counter()-started,
        H_estimator='exact' if h_samples_per_root is None else 'unbiased_abs_weighted_connections',
        H_samples_per_root=h_samples_per_root,K_estimator='exact_combined_Lminus_Lplus_plus_M_Mplus1',
        root_distribution='exactly_uniform_fixed_Ne_M_with_replacement',
        explicit_many_body_basis_enumeration=False,reference_data_loaded=False)
