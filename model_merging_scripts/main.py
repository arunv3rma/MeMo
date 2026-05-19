import argparse
import os

from model_merge.model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple AI models")
    parser.add_argument(
        "--models", nargs="+", required=True, help="Paths to model directories to merge"
    )
    parser.add_argument(
        "--method",
        choices=["linear", "slerp", "task", "ties", "dare_ties", "dare_linear", "fusion", "fusion_iqr", "fusion_knee"],
        required=True,
        help="Merge method to use.",
    )
    parser.add_argument(
        "--weight",
        default=None,
        type=float,
        nargs="+",
        help="Weights for linear interpolation. Must match number of models.",
    )
    parser.add_argument(
        "--density",
        default=0.5,
        type=float,
        nargs="+",
        help="Density factors for task arithmetic sparsification. Must match number of models.",
    )
    parser.add_argument(
        "--t",
        default=0.5,
        type=float,
        help="Interpolation factor for SLERP (between 0 and 1). Required when method is 'slerp' (default: 0.5).",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base model path for task arithmetic methods. Required when using task-based methods.",
    )
    parser.add_argument(
        "--output",
        default="./merged",
        required=False,
        help="Output directory for merged model",
    )

    args = parser.parse_args()

    # Validate SLERP parameters
    if args.method == "slerp":
        if args.t is None:
            parser.error("--t is required when using SLERP")
        if not 0 <= args.t <= 1:
            parser.error("--t must be between 0 and 1")
        if args.weight is not None:
            parser.error("--weight cannot be used with SLERP")
        if args.density is not None:
            parser.error("--density can only be used with task arithmetic")
    # Validate Fusion parameters
    elif args.method == "fusion" or args.method == "fusion_iqr" or args.method == "fusion_knee":
        if args.t is None:
            parser.error("--t is required when using Fusion")
        if args.weight is not None:
            parser.error("--weight cannot be used with Fusion")
    else:
        if args.density:
            # Validate density values
            if len(args.density) != len(args.models):
                parser.error("Number of density values must match number of models")
            if not all(0 <= d <= 1 for d in args.density):
                parser.error("Density values must be between 0 and 1")

    # Validate task arithmetic parameters
    if args.method in ["task", "ties", "dare_ties", "dare_linear"]:
        if args.base is None:
            parser.error("--base is required when using task arithmetic methods")
        if not os.path.exists(args.base):
            parser.error(f"Base model path does not exist: {args.base}")
        if args.density:
            # Validate density values
            if len(args.density) != len(args.models):
                parser.error("Number of density values must match number of models")
            if not all(0 <= d <= 1 for d in args.density):
                parser.error("Density values must be between 0 and 1")

    # Validate and prepare output directory
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=== Model Merge ===")

    print("\n[1/3] Loading source models")
    model = Model(
        args.models, method=args.method, output_dir=output_dir, base_model=args.base
    )

    print("\n[2/3] Merging models")
    model.merge(t=args.t, weights=args.weight, density=args.density)

    print("\n[3/3] Saving merged model")
    output_dir = model.save()
    print("Complete!")
