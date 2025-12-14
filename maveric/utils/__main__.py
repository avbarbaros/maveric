"""Allow running utils modules as: python -m maveric.utils.MODULE_NAME"""
import sys

if len(sys.argv) < 2:
    print("Usage: python -m maveric.utils.MODULE_NAME [args...]")
    print("Available modules:")
    print("  balance_cli - Balance manually cleaned training datasets")
    sys.exit(1)

module_name = sys.argv[1]
sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove module name from args

if module_name == 'balance_cli':
    from .balance_cli import main
    main()
else:
    print(f"Unknown module: {module_name}")
    sys.exit(1)
