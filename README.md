# How to run project

## Structure folder

This project is organized into two main folders:

- **`envs/`**: Contains environment for machine learning.
- **`apps/`**: Contains frontend, backend and machine learning app.

## Requirements

- Node 20+, pnpm 10+
- Conda 23+

## Install dependencies

```bash
pnpm install
```

## Activate conda environment

Create and activate the conda environment from the yml file:

```bash
conda env create -f envs/datathon2025.yml
conda activate datathon2025
```

## Run project

### Run all

```bash
pnpm dev
```

### Only server

```bash
pnpm be dev
```

### Only web

```bash
pnpm fe dev
```

## Outcome

Web: https://localhost:4000
Server: https://localhost:5000
