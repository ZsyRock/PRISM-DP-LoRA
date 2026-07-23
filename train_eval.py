from __future__ import annotations
import argparse
import sys
from dataclasses import fields
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
from prism_cli.eval_glue import evaluate_glue8
from prism_cli.eval_math import evaluate_math10k
from prism_cli.experiment_identity import load_json_object
from prism_cli.trainers import RunConfig, train

def parse_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).lower()
    if s in {'1', 'true', 'yes', 'y'}:
        return True
    if s in {'0', 'false', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected boolean, got {x!r}')

CONFIG_KEY_ALIASES = {
    'epsilon': 'dp_epsilon',
    'delta': 'dp_delta',
    'steps': 'total_update_steps',
    'lr': 'learning_rate',
    'initial_clip_threshold': 'dp_max_grad_norm',
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train and evaluate PRISM LoRA adapters on Math-10K or GLUE8.')
    p.add_argument('--config', type=Path, default=None, help='JSON defaults; explicitly supplied CLI options take precedence.')
    p.add_argument('--dataset', choices=['math10k', 'math', 'glue8', 'glue'], default=None)
    p.add_argument(
        '--method',
        choices=['baseline', 'slaclip', 'slaclip_q'],
        default='baseline',
        help='Fixed PRISM, official full SlaClip+PRISM, or fixed-target SlaClip-Q+PRISM.',
    )
    p.add_argument('--privacy', choices=['dp', 'nondp', 'non-dp'], default='dp')
    p.add_argument('--epsilon', dest='dp_epsilon', type=float, default=6.0)
    p.add_argument('--delta', dest='dp_delta', type=float, default=1e-05)
    p.add_argument('--base_model', default='google/gemma-3-4b-pt')
    p.add_argument('--model_revision', default='main', help='Hugging Face model revision; pin a commit for formal runs.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run_name', default=None, help='Human-readable label; a config hash is always appended.')
    p.add_argument('--repeat_id', type=int, default=None)
    p.add_argument('--data_path', type=Path, default=None)
    p.add_argument('--output_dir', type=Path, default=None)
    p.add_argument('--result_dir', type=Path, default=None)
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--target_modules', default='q_proj,k_proj,v_proj,up_proj,down_proj')
    p.add_argument('--steps', dest='total_update_steps', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--micro_batch_size', type=int, default=4)
    p.add_argument('--lr', dest='learning_rate', type=float, default=None)
    p.add_argument('--cutoff_len', type=int, default=None)
    p.add_argument('--train_on_inputs', type=parse_bool, default=None)
    p.add_argument(
        '--initial_clip_threshold',
        '--dp_max_grad_norm',
        dest='dp_max_grad_norm',
        type=float,
        default=1.0,
        help='Fixed C for baseline; initial C0 for full SlaClip. --dp_max_grad_norm is a legacy alias.',
    )
    p.add_argument('--dp_grad_sample_mode', default='functorch')
    p.add_argument('--dp_accountant', choices=['rdp', 'prv', 'gdp'], default='prv')
    p.add_argument('--dp_secure_mode', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--require_cuda', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--telemetry_mode', choices=['dp_safe', 'research_raw'], default='dp_safe')
    p.add_argument('--allow_non_private_telemetry', action=argparse.BooleanOptionalAction, default=False, help='Required acknowledgement for exact, non-DP research diagnostics.')
    p.add_argument('--raw_hist_bins', type=int, default=32)
    p.add_argument('--raw_hist_max', type=float, default=0.0, help='Fixed norm histogram upper edge; 0 uses 4 * initial C.')
    p.add_argument('--slaclip_num_slots', type=int, default=0, help='Slack dimension K; 0 selects the paper bound automatically.')
    p.add_argument('--slaclip_eta', type=float, default=0.5)
    p.add_argument('--slaclip_beta', type=float, default=0.5)
    p.add_argument(
        '--slaclip_target_clip_fraction',
        type=float,
        default=0.99,
        help=(
            'Requested clipped fraction for slaclip_q. Internally gamma is the '
            'complementary target unclipped-CDF proxy; 0.99 therefore maps to gamma=0.01.'
        ),
    )
    p.add_argument('--slaclip_c_min', type=float, default=0.1)
    p.add_argument('--slaclip_c_max', type=float, default=50.0)
    p.add_argument('--run_train', type=parse_bool, default=True)
    p.add_argument('--run_eval', type=parse_bool, default=True)
    p.add_argument('--force_train', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--force_eval', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--no_resume', dest='resume', action='store_false', help=argparse.SUPPRESS)
    p.add_argument('--checkpoint_every', type=int, default=25)
    p.add_argument('--eval_batch_size', type=int, default=None)
    p.add_argument('--num_beams', type=int, default=None)
    p.add_argument('--max_new_tokens', type=int, default=None)
    p.add_argument('--max_input_length', type=int, default=None)
    p.add_argument('--fast_dev_run', type=int, default=0)
    return p


def _json_defaults(config_path: Path, parser: argparse.ArgumentParser) -> dict:
    payload = load_json_object(config_path)
    schema_version = payload.pop('schema_version', 1)
    if schema_version != 1:
        parser.error(f'Unsupported config schema_version={schema_version!r}; expected 1')
    valid_destinations = {action.dest for action in parser._actions}
    valid_destinations.update(item.name for item in fields(RunConfig) if item.init and item.name != 'root')
    defaults = {}
    sources = {}
    for key, value in payload.items():
        destination = CONFIG_KEY_ALIASES.get(key, key)
        if destination not in valid_destinations or destination in {'help', 'config'}:
            parser.error(f'Unknown JSON config field: {key}')
        if destination in defaults and defaults[destination] != value:
            parser.error(
                f'Conflicting JSON config fields {sources[destination]!r} and {key!r} '
                f'both map to {destination!r}'
            )
        defaults[destination] = value
        sources[destination] = key
    return defaults


def parse_cli_args(argv=None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=Path, default=None)
    preliminary, _ = pre_parser.parse_known_args(argv)
    parser = build_parser()
    if preliminary.config is not None:
        parser.set_defaults(**_json_defaults(preliminary.config, parser))
    args = parser.parse_args(argv)
    if args.dataset is None:
        parser.error('--dataset is required either on the CLI or in --config')
    for name in ('data_path', 'output_dir', 'result_dir'):
        value = getattr(args, name)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(value))
    if isinstance(args.target_modules, str):
        args.target_modules = [x.strip() for x in args.target_modules.split(',') if x.strip()]
    elif isinstance(args.target_modules, list) and all(isinstance(x, str) for x in args.target_modules):
        args.target_modules = [x.strip() for x in args.target_modules if x.strip()]
    else:
        parser.error('target_modules must be a comma-separated string or a JSON list of strings')
    return args


def main(argv=None):
    args = parse_cli_args(argv)
    run_fields = {item.name for item in fields(RunConfig) if item.init and item.name != 'root'}
    cfg_values = {name: getattr(args, name) for name in run_fields if hasattr(args, name)}
    cfg = RunConfig(root=ROOT, **cfg_values).finalize()
    if cfg.run_train:
        train(cfg)
    else:
        print('[skip] run_train=False')
    if cfg.run_eval:
        if cfg.dataset == 'math10k':
            evaluate_math10k(cfg, batch_size=args.eval_batch_size or 64, num_beams=args.num_beams or 4, max_new_tokens=args.max_new_tokens or 256, max_input_length=args.max_input_length or 1024)
        else:
            evaluate_glue8(cfg, batch_size=args.eval_batch_size or 128, num_beams=args.num_beams or 1, max_new_tokens=args.max_new_tokens or 8, max_input_length=args.max_input_length or 384, fast_dev_run=args.fast_dev_run)
    else:
        print('[skip] run_eval=False')
if __name__ == '__main__':
    main()
