"""Allow running the package with `python -m calcium_analysis`."""


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="calcium-analysis", description="calcium_analysis command-line utility"
    )
    parser.add_argument("--version", action="store_true", help="Show package version")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("show-modules", help="List modules inside the package")

    args = parser.parse_args()

    if args.version:
        try:
            from importlib.metadata import version

            print(version("calcium_analysis"))
        except Exception:
            print("calcium_analysis (not installed as a package)")
        return

    if args.command == "show-modules":
        import pkgutil
        import calcium_analysis

        print("Modules under `calcium_analysis`:")
        for _, name, _ in pkgutil.iter_modules(calcium_analysis.__path__):
            print("-", name)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
