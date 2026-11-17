"""Portable VMC FQHE entrypoint with unique fresh outputs and strict identity."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,uuid
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'shared'),str(ROOT/'shared/vendor/jaqmc_spring/src')]
from fqh_spring.train import main,load_config
if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'configs/ne4.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    args,_=parser.parse_known_args()
    config=load_config(args.config)
    if config['method']!='vmc_sgd':raise ValueError('VMC entrypoint requires method=vmc_sgd')
    help_requested=any(arg in ('-h','--help')for arg in sys.argv[1:])
    if args.resume and args.output is None and not help_requested:
        raise ValueError('Resume requires explicit --output pointing to the existing run')
    if not args.resume and args.output is None and not help_requested:
        tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        output=Path(__file__).resolve().parents[1]/'results/runs'/f"ne{config['ne']}_vmc_sgd_{tag}"
        sys.argv[1:1]=['--output',str(output)]
    if not any(arg=='--config' or arg.startswith('--config=')for arg in sys.argv[1:]):
        sys.argv[1:1]=['--config',str(args.config)]
    main()
