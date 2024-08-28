# UDP Message Transmission and Verification

This project provides a set of Python scripts for securely transmitting and verifying UDP messages. The system breaks up a text message into blocks, applies HMAC (Hash-based Message Authentication Code) for integrity and authenticity verification, and sends these blocks over a network using UDP. The receiver script listens for these messages, verifies their authenticity, and provides useful analysis.

## Table of Contents

- [UDP Message Transmission and Verification](#udp-message-transmission-and-verification)
  - [Table of Contents](#table-of-contents)
  - [Files](#files)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
    - [Running the Sender](#running-the-sender)
    - [Running the Receiver](#running-the-receiver)
  - [How It Works](#how-it-works)
    - [Sender Script](#sender-script)
    - [Receiver Script](#receiver-script)
  - [Analysis](#analysis)
  - [License](#license)

## Files

- `sender.py`: Script to send the UDP messages with HMAC-based integrity checks.
- `receiver.py`: Script to receive and verify the UDP messages.
- `README.md`: This file, which explains the usage and functionality.

## Prerequisites

- Python 3.x
- `numpy` library: Install using `pip install numpy`

## Usage

### Running the Sender

1. **Prepare the Text File:**
   - Ensure you have a text file named `send.txt` in the same directory as the `sender.py` script. This file should contain the text you wish to transmit.

2. **Run the Sender Script:**
   - Execute the sender script using the following command:
     ```bash
     python sender.py
     ```
   - The script will split the text into blocks, compute HMACs, and send the messages over UDP to the specified IP address and port.

### Running the Receiver

1. **Run the Receiver Script:**
   - Execute the receiver script using the following command:
     ```bash
     python receiver.py
     ```
   - The receiver will listen on the specified IP address and port for incoming UDP messages. It will verify the authenticity of each message using the HMAC and provide a summary analysis of the results.

## How It Works

### Sender Script

The sender script (`sender.py`) performs the following steps:

1. **Divide the Text:**
   - The text is divided into blocks of a specified size (`block_size_bits`). Padding is added to ensure the number of blocks is compatible with the matrix `X`.

2. **Generate HMAC:**
   - For each block, an HMAC is generated using the key `b"key"` and the SHA-384 algorithm. The HMAC ensures the integrity and authenticity of the message.

3. **Send Messages:**
   - The blocks, along with their HMACs, are sent as UDP packets to the specified IP address and port. A delay is introduced between each transmission to simulate real-world conditions.

4. **Attack Simulation:**
   - The script can randomly drop certain messages to simulate packet loss or attack scenarios.

### Receiver Script

The receiver script (`receiver.py`) performs the following steps:

1. **Listen for Messages:**
   - The receiver listens for incoming UDP messages on the specified IP address and port.

2. **Verify HMAC:**
   - Upon receiving a message, the script extracts the Frame Number (FN) and the block of data. It then verifies the HMAC to ensure the message's authenticity.

3. **Analyze Results:**
   - The script tracks how many messages were verified successfully and how many were not. It provides a summary of the verification success rate.

## Analysis

After the transmission, the receiver script provides a summary of the message verification process, including:

- Total number of messages received.
- Number of messages successfully verified.
- Number of messages that failed verification.
- Verification success rate as a percentage.

This analysis helps in understanding the reliability and security of the message transmission process.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

This `README.md` should help users understand the purpose of the scripts, how to set them up, and how to run them effectively.