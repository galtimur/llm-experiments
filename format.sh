
exception_folder="optimization_trajectory"

folders=$(find . -type d -not -path "./$exception_folder/*" -not -path "./$exception_folder" -not -path "./.*" -not -path ".")
files=$(find . -mindepth 1 -maxdepth 1 -type f -not -path "./.*" -not -path "./*.sh")
targets=$(echo "$folders"; echo "$files")

echo "-------------- Running Black... --------------"
black $targets

echo "-------------- Running iSort... --------------"
isort $targets

echo "-------------- Running Ruff... --------------"
ruff check $targets

echo "-------------- Running MyPy... --------------"
mypy $targets

echo "All linters and type checkers have been run."