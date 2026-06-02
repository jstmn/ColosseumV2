black --line-length 120 diffusion_policy/*.py
black --line-length 120 *.py
ruff check --fix .
ruff check --fix diffusion_policy/