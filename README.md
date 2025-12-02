# AlpaSim: A modular, lightweight, and data-driven research simulator for autonomous driving
<div align="center">
  <img src="docs/assets/images/thumbnail.gif" alt="AlpaSim Simulation Demo" width="600">
</div>


## What is AlpaSim?

AlpaSim is an open-source autonomous vehicle simulation platform designed specifically for research
and development. It allows users to test end-to-end AV policies in a closed-loop setting by
simulating realistic sensor data, vehicle dynamics, and traffic scenarios within a modular and
extensible testbed.

Suitable use cases include:

- **Algorithm Validation**: Test new autonomous driving algorithms in realistic environments
- **Safety Analysis**: Evaluate vehicle behavior in edge cases and challenging scenarios
- **Performance Benchmarking/Regression Testing**: Compare different models and configurations
- **Debugging**: Understand and debug complex autonomous driving behaviors
<!-- TODO: Should we add notes on RL/data generation? -->


### **Sensor Fidelity**
- Neural Rendering (NuRec) integration for photorealistic sensor simulation of novel views
- High-fidelity camera feeds with configurable field-of-view, resolution, and frame rates
- Realistic sensor noise and environmental conditions

### **Research Hackability**
- Python-based implementation built for rapid prototyping and experimentation
- Modular grpc interface design allows researchers to swap out components with custom implementations
- Extensive configuration options and debugging tools

### **Horizontal Scalability**
- Microservices architecture enabling distributed computing
- Scale individual components for optimal load balancing
- Support for multi-node deployments

To learn more about the design principles and architecture, check out the [system design docs](docs/DESIGN.md).


## Documentation & Resources

- **[Onboarding Guide](docs/ONBOARDING.md)**: Initial setup and access instructions
- **[Tutorial](docs/TUTORIAL.md)**: Step-by-step usage guide
- **[Operations Guide](docs/OPERATIONS.md)**: Performance tuning, configuration, and troubleshooting
- **[Design Documentation](docs/DESIGN.md)**: Technical architecture and design decisions
- **[API Reference](src/grpc/)**: gRPC API documentation

### **Sample Data**
- **Hugging Face Dataset**: [PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)
- **Sample Artifacts**: Included in the repository via Git LFS


## Contributing

We welcome contributions from the research community! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and conventions
- Testing requirements
- Pull request process
- Development setup


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation
If you use this software, please cite it as follows:

```
@software{alpasim_2025,
  author       = {
    NVIDIA and
    Yulong Cao and
    Riccardo de Lutio and
    Sanja Fidler and
    Guillermo Garcia Cobo and
    Zan Gojcic and
    Maximilian Igl and
    Boris Ivanovic and
    Peter Karkus and
    Janick Martinez Esturo and
    Marco Pavone and
    Aaron Smith and
    Ellie Tanimura and
    Michal Tyszkiewicz and
    Michael Watson and
    Qi Wu and
    Le Zhang
  },
  title        = {AlpaSim: A Modular, Lightweight, and Data-Driven Research Simulator for Autonomous Driving},
  year         = {2025},
  month        = {October},
  url          = {https://github.com/NVlabs/alpasim},
}
```

## Project Contributors:
Contributors in each topic in alphabetical order

**Project Lead:** Maximilian Igl

**Tech Leads:** Michal Tyszkiewicz, Michael Watson

**Architecture Design & Networking:** Michal Tyszkiewicz

**Open Sourcing:** Guillermo Garcia Cobo, Maximilian Igl, Peter Karkus, Ellie Tanimura, Michael Watson

**Infrastructure & Wizard:** Maximilian Igl, Aaron Smith, Michal Tyszkiewicz, Michael Watson, Qi Wu (SLURM deployment), Le Zhang (Data management)

**Runtime:** Maximilian Igl, Aaron Smith, Ellie Tanimura, Michal Tyszkiewicz, Michael Watson

**CICD:** Maximilian Igl, Aaron Smith

**Data Pipeline:** Riccardo de Lutio, Janick Martinez, Le Zhang

**Product Manager:** Matt Cragun

**Testing & debugging:** Guillermo Garcia Cobo, Peter Karkus, Ellie Tanimura

**Service Modules:**
* Driver integration: Maximilian Igl, Peter Karkus, Michal Tyszkiewicz
* Evaluation: Yulong Cao, Maximilian Igl
* Controller: Michael Watson
* Physics: Riccardo de Lutio
* Trafficsim: Maximilian Igl, Boris Ivanovic

**Senior Mgmt:** Sanja Fidler, Zan Gojcic, Boris Ivanovic, Marco Pavone

**Acknowledgements for additional contributions:** Fabian Barajas, Kashyap Chitta, Ankit Gupta, Laura Leal-Taixe, Nicole Yang

<div align="center">
  <strong>Built for researchers, by researchers</strong><br>
  <em>Accelerating autonomous vehicle development through realistic simulation</em>
</div>
