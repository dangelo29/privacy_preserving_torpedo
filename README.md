# Thesis Link:

https://repositorio.ul.pt/bitstream/10451/56882/1/TM_Daniel_Angelo.pdf

# How to Run:

- Main program for the ranking stage: `server-ranking.py`
- Main program for the correlation stage: `server-correlation.py`
- Main client program: `client.py`

The server scripts are essential for the solution to function as they enable ISPs to participate in the distributed computations and act as TF Encrypted parties. Each script can be represented as follows, where "player" corresponds to the name of the party (e.g., “server0”) and "config" is a file that has the hostmap configuration detailed. in order to start the parties, we execute this commmand:

```
python3 -m tf_encrypted.player {player} --config {config}
```

The `config.json` file specifies the IP addresses of the participants and the ports from where they communicate with each other:

```
{
    "server0": "10.154.0.3:4440",
    "server1": "10.164.0.4:4440",
    "server2": "10.132.0.10:4440",
    "operator": "10.186.0.2:4440"
}
```

# Requirements:

This project was executed with the following dependencies:

- Python 3.6.9
- TensorFlow 1.15.5
- Keras 2.2.4-tf
- NumPy 1.18.5
- Matplotlib 3.3.4
- TensorFlow Encrypted 0.7.0
- TensorFlow Privacy 0.0.1
