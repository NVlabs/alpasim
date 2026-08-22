# SimScale LTF Source Notice

This package is an inference-only adaptation of the Apache-2.0 SimScale Latent TransFuser model. Training, dataset, nuPlan, NAVSIM, Hydra, and Lightning dependencies have been removed. The timm encoder is constructed with `pretrained=False`, and the service is considered ready only after the complete checkpoint has been loaded strictly.

| Source path | SHA256 |
| --- | --- |
| `reference/SimScale/navsim/agents/transfuser/transfuser_config.py` | `7c10f21a374c3c9ad99fb1619a1d12c62487c9d30f2e98b0e4dc127785861f57` |
| `reference/SimScale/navsim/agents/transfuser/transfuser_backbone.py` | `5563645725c3795e621cb48bc5ee075fd93afe25b6b2f8203703f7977fa2814e` |
| `reference/SimScale/navsim/agents/transfuser/transfuser_model.py` | `7b2c996b8f4964c30b66eae0e972988f70c5b40732352713c624c40065801378` |
| `reference/SimScale/navsim/agents/transfuser/transfuser_features.py` | `26c94afb5841f5c1a76ffd634c57f985f5b68be32413cda2266d7a754af55629` |
| `reference/SimScale/navsim/agents/transfuser/transfuser_agent.py` | `35d42e03ee1ffdda88c1b1ebfe327748172d67ef125fd48be2787bc06fb409cc` |
| `reference/SimScale/LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
