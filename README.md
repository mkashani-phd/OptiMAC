# OptiMAC: Robust Optimization Framework for MAC Aggregation in Adversarial Environments

OptiMAC is a novel optimization framework designed to enhance the integrity and efficiency of Message Authentication Code (MAC) aggregation schemes. By systematically optimizing tag-to-message assignments, OptiMAC achieves superior performance, balancing security and efficiency in wireless communication networks under adversarial conditions.

---

## Key Features

- **Graph-Based Dependency Mapping**: Models message-to-tag dependencies as a graph to enable systematic optimization.
- **Simulation Tools**: MATLAB and Python tools for analyzing different MAC configurations.
- **Real-World Demonstration**: Includes a UDP-based demo for WiFi and 5G networks with jamming resilience.
- **Custom Metrics**: Includes Goodput, Tag-bits per Message (TbpM), and Strength Number (SN) for evaluating performance.

---

## Repository Structure

### Core Directories

- **`img/`**: Contains documentation diagrams and visualizations.
  - `MAC.drawio`, `ProMAC_Graph.ipynb`: Illustrations of MAC dependencies.
- **`Optimizer/`**: Implements the optimization framework.
  - `RunOptimizer.ipynb`: Main optimization script that runs `optimize.py` in terminal.
  - `Strength_Number.ipynb`: Notebooks for calculating the strength number using pulp optimizer.
  - `Xs.pkl`: A dictionary data structure to save the optimizer's results.
  - `TagModel.lp`: Pre-computed optimization model (created and used internally by Gurobi).
- **`Simulation/`**: Tools for analyzing MAC configurations.
  - `2D_MAC.m`: MATLAB script for 2D MAC analysis.
  - `Simulation.ipynb`: Python notebook for simulation of the `Xs.pkl` with random packet losses.
  - `time_series.csv`, `time_series2.csv`: Sample simulation packet loss recorded from real 5G communication.
- **`UDP Demo/`**: Real-world implementation of OptiMAC using WiFi and 5G.
  - **`Blockwise/`**: Traditional MAC implementation for the webcam live streaming.
  - **`SDR jammer/`**: Scripts for Software Defined Radio (SDR)-based jamming experiments.
  - `rx.py`, `tx.py`: Receiver and transmitter implementations.
  - `rx.ipynb`, `tx.ipynb`: Receiver and transmitter code that run the `rx.py` and `tx.py` on terminal.
  - `results_WiFi.pkl`, `results_5G.pkl`: Experimental results for WiFi and 5G.
- **`src/`**: Python package with reusable modules.
  - `Auth.py`: Basic metric calcultion used for static analysis and simulations.
  - `TagModel.py`: Core implementation of tag-to-message assignment optimization.
  - `TagModel_lat.py`: An attempt to optimize the security and latency by adding the two objective function with coefficients. 
  - `model_markdown.ipynb`: Documentation behind the optimizer model.

### Supporting Files
- **`.gitignore`**: Specifies files to exclude from version control.
- **`CITATION.cff`**: Citation details for referencing this work.
- **`README.md`**: Overview and instructions (this file).

---

## Installation

### Prerequisites
- **Python 3.9+**
- **MATLAB (Optional)** for simulation scripts.
- Required Python libraries: Install using the provided `requirements.txt`.

```bash
pip install -r requirements.txt
```
You also need the openCV and the Gurobi if you want to runt the UDP and run the optimizer, respectively.

### Setup
Clone the repository with submodules that support the UDP demo test:
```bash
git clone --recurse-submodule git@github.com:mkashani-phd/optiMAC.git
cd optiMAC
```

---

## Usage

### Optimization
1. Navigate to the `Optimizer/` directory.
2. Modify the parameters in `optimizer.ipynb` or `TagModel.lp`.
3. Run the script:
   ```bash
   python optimize.py
   ```

### Simulation
1. Use `Simulation/2D_MAC.m` in MATLAB for custom analysis.
2. Alternatively, run `Simulation.ipynb` in Python for simulation-based exploration.

### UDP Demo
1. Configure WiFi or 5G parameters in the `UDP Demo/` directory.
2. Transmit data using:
   ```bash
   python tx.py
   ```
3. Receive data using:
   ```bash
   python rx.py
   ```

---

## Experimental Setup

### WiFi Testbed
- **Hardware**: WiFi hotspot, Logitech HD Pro Webcam
- **Software**: Linux

### 5G Testbed
- **Hardware**: USRP B210 SDRs, OpenAirInterface, Open5GS
- **Software**: Ubuntu 22.04 LTS
- **Protocol**: 5G Core Network

---

## Results

OptiMAC achieves:
- Enhanced security by increasing **Tag-bits per Message (TbpM)** at lower **tag-to-message ratio**.
- Higher **Goodput** at low **tag-to-message ratio**.
- Resilience against adversarial attacks, such as jamming and DoS.

Experimental results are available in the `Analyzing_Results.ipynb` files within `UDP Demo`.

---

## Contributing

We welcome contributions! Please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References

1. Armknecht et al., "Progressive MACs for Continuous Efficient Authentication of Message Streams," CCS '20.
