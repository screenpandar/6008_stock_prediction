"""Project experiment entry point / 项目统一实验入口。"""
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        from scripts.evaluate_test import main as test_main
        test_main(sys.argv[2:])
    else:
        from scripts.run_experiment import main as validation_main
        validation_main()


if __name__ == "__main__":
    main()
