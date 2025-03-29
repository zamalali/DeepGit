import pkg_resources

with open("requirements.txt", "r") as f:
    packages = [line.strip() for line in f if line.strip()]

with open("requirements.txt", "w") as f:
    for pkg in packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            f.write(f"{pkg}=={version}\n")
        except Exception:
            f.write(f"{pkg}\n")  # leave unpinned if not found
