# CoolLoad
## Inspiration

Artificial Intelligence is driving the most significant technological leap of our generation. However, this incredible progress comes with a physical cost: massive computational power generates immense heat. As we train and fine-tune increasingly complex models, data centers work in overdrive, often exacerbating the "heat island" effect and straining local cooling infrastructure.

We firmly believe that halting AI development is not the answer. When humanity realized that gasoline cars were polluting our air, we didn't ban transportation; we embraced the problem and engineered hybrids and electric vehicles to protect our environment. We must apply that same spirit of innovation to AI.

This led us to ask:
- What if we could democratize this eco-conscious approach and dynamically redistribute heavy AI workloads across the globe based on real-time weather patterns?
- What if data centers in naturally cooler, breezier climates could take on the heavy lifting during peak heat waves elsewhere, actively preventing the exacerbation of heat islands?
- And what if every company - from startups to enterprise giants - could train the models of tomorrow without overheating the cities of today?

## What It Does

Our platform dynamically orchestrates and redistributes computationally heavy AI workloads - such as model training, fine-tuning, generational tasks - to data centers positioned in optimal environmental conditions. By leveraging localized, real-time climate data, the system prevents data centers from curbing the creation of heat islands and blocking massive thermal waves from sweeping into surrounding residential communities.

## Architecture

We combine the reasoning of Gemma 4 with our physical ML model. Together, they form a closed-loop system that ingests raw code and environmental telemetry, simulates thermodynamic impacts, and intelligently distributes workloads across a global network of data centers.

**Gemma 4**
- Gemma 4 serves as the intelligent backbone of our architecture, operating across five distinct roles to handle data ingestion, code analysis, orchestration, edge communication, and reporting.
- Data parsing and geographic grounding: When a data center joins the network, Gemma 4 parses its raw technical specifications. Using the data center’s physical address, the model automatically constructs a structured query to the OpenStreetMap API and extracts exact latitude, longitude, and geographic boundary polygons. This ensures all downstream physical simulations are precisely grounded in real-world geography.
- Computational load estimation: Before code execution begins, Gemma 4 analyzes the user's training script or fine-tuning pipeline. By evaluating model size, batch sizes, and epoch counts, it estimates the total power load required to run the code, represented in Megawatts (MW). If Gemma faces any uncertainties, it will ask the user specific questions to verify assumptions. For simplification, we consider all the data centers use the same GPU - H100.
- Global orchestrator (decision maker): Acting as the central brain, the Gemma 4 Orchestrator synthesizes data from the physical model, current grid status, and available data center capacities to make the final executive decision on where workloads should be routed.
- Local edge agents: Every data center in our network runs a localized Gemma 4 agent. These agents maintain a real-time heartbeat with the central Orchestrator - communicating local telemetry, flagging operational anomalies, verifying routing decisions, and issuing local environmental alerts if unexpected weather changes occur.
- Report generation: Post-execution, Gemma 4 aggregates energy consumption profiles and regional temperature deltas to generate comprehensive, human-readable environmental impact reports, detailing the exact amount of thermal pollution avoided.

**Physical model**
Physical model handles the complex thermodynamics of heat dissipation. It simulates how heat moves from a data center building into the atmosphere and surrounding communities to find the mathematically optimal workload distribution. To build an accurate thermodynamic simulation, the physical model ingests two primary streams of data:
1. Weather data: Real-time atmospheric conditions including wind speed/direction, air temperature, and localized humidity.
2. Building data: The total area, the total compute load, and the thermal capacity of the data center's construction materials (defining how the building absorbs and radiates heat).
Model outputs dynamic heatmap distributions - a thermodynamic simulation mapping exactly how heat will dissipate into the surrounding environment for each participating data center - and recommended load per data center: an optimized workload allocated to each facility to prevent the formation of heat islands.

## How To Run
### Quick Run On Kaggle (Test Gemma 4)
Ensure that sample data is added to the input directory (data is available in `input_data_sample/` also public on Kaggle):
* ML_code: example code (Gemma fine-tuning).
* DC_specs: PDF files containing data center building specifications, including a table of heat capacities per material.

Open the Kaggle notebook `Gemma4Hackathon.ipynb`. This notebook loads Gemma 4 with Unsloth. Click **Run All** to execute the workflow using the default sample data. The notebook will output:

1. Parsed and organized data center specifications.
2. Estimated power load for the provided sample code.
3. Example of local agent verdicts – indicating whether a workload is accepted or rejected, along with reasoning that is communicated back to the orchestrator.
### Full Project Run
_Install python dependencies._
```
pip install -r requirements.txt
```
#### Start Simulation Server
Add the following to your `.env` file:
```
GOOGLE_STUDIO_AI_API_KEY=xxx
LOCAL_AGENT_MODEL=gemma-4-31b-it
```
The command below will use Gemma 4 via the Google API (for a simplified deployement):
```
uvicorn simulation_api_server:app --host 127.0.0.1 --port 8765
```

#### Run Frontend App
```
cd frontend/dc-heat-dashboard
npm install
npm run dev
```
Access the website: `http://localhost:5173/`
