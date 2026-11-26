"""Portable SPRING FQHE entrypoint with unique fresh outputs and strict identity."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys,uuid
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'shared'),str(ROOT/'shared/vendor/jaqmc_spring/src')]
from fqh_spring.train import main,load_config

def default_config(output, resume):
    configs = Path(__file__).resolve().parent / 'configs'
    if resume and output is not None:
        metadata = output / 'run_config.json'
        if metadata.is_file():
            mode = json.loads(metadata.read_text())['configuration'].get('amplitude_map', 'exponential')
            if mode == 'exponential':
                return configs / 'ne4.json'
    return configs / 'ne4_cartesian.json'

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    args,_=parser.parse_known_args()
    config_path = args.config or default_config(args.output, args.resume)
    config=load_config(config_path)
    if config['method']!='spring':raise ValueError('SPRING entrypoint requires method=spring')
    help_requested=any(arg in ('-h','--help')for arg in sys.argv[1:])
    if args.resume and args.output is None and not help_requested:
        raise ValueError('Resume requires explicit --output pointing to the existing run')
    if not args.resume and args.output is None and not help_requested:
        tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        output=Path(__file__).resolve().parents[1]/'results/runs'/f"ne{config['ne']}_spring_{tag}"
        sys.argv[1:1]=['--output',str(output)]
    if not any(arg=='--config' or arg.startswith('--config=')for arg in sys.argv[1:]):
        sys.argv[1:1]=['--config',str(config_path)]
    main()
