# SPARKLE 

## Setting up

### Installation 
```shell
# Install uv (the fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh 

# or see: https://docs.astral.sh/uv/

# Verify installation
uv --version
```

### Project Setup 

```shell
uv sync
```

### Training the model

```shell
uv python -m sparkle.model.scripts.train
```

## Contributors

1. Satvik (https://github.com/satvik13o7/)
2. Kavish Kashyap 
3. Arnav Garg (https://github.com/dorkcubed/)
4. Atharva Gupta (https://github.com/atharva7-g/)


### Notes

`manifest.json` format
```json
[
    {
        "flow": "flow_001",
        "packet": "s3://netml-s3-bucket/.../packets/0001.txt",
        "header": "s3://netml-s3-bucket/.../header/0001.txt",
        "field": "s3://netml-s3-bucket/.../fields/0001.txt",
        "direction": "s3://netml-s3-bucket/.../direction/0001.txt"
    },
    {
        "flow": "flow_002",
        "packet": "s3://netml-s3-bucket/.../packets/0002.txt",
        "header": "s3://netml-s3-bucket/.../header/0002.txt",
        "field": "s3://netml-s3-bucket/.../fields/0002.txt",
        "direction": "s3://netml-s3-bucket/.../direction/0002.txt"
    }
]
```

```
Total: 239223
Train: 203339 | Val: 17941 | Test: 17943
```