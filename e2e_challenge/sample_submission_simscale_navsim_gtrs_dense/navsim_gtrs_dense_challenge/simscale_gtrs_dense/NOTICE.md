# SimScale GTRS-Dense inference adaptation

This directory contains an inference-only adaptation of SimScale GTRS-Dense,
licensed under the adjacent Apache-2.0 `LICENSE`.

The ResNet34 and latent LiDAR Transfuser graph and the optional VoV V-99-eSE
graph are adapted from these upstream sources:

| Source | SHA-256 |
| --- | --- |
| `gtrs_agent.py` | `6e5c96b43c4729b09a282fb317dc9afc48402980aa71a91b799e03c858c22398` |
| `hydra_config.py` | `25789c7438e18d1f6a103dd0624813a7ceac048a899c0ed1c8233980f8118898` |
| `hydra_model.py` | `94205256d25025e9f035b0d3f5d37eb0979213aa5e4832f6d88b97277dbea7e5` |
| `transfuser_backbone.py` | `09cf91ba32a42443f14c44e9ac8630831ca537064b5201958736c075f153b89a` |
| `utils/attn.py` | `9ad0f145469a257d59fac15db4847b24882c7046e9a651496f19c54c1f38708f` |
| `transfuser_model.py` | `7b2c996b8f4964c30b66eae0e972988f70c5b40732352713c624c40065801378` |
| `backbones/vov.py` | `8c43d93743ba977c987c2abdc4bb0ab76394371e4e7695febc94b547b7bd2b2f` |

Adaptations remove training and dataset dependencies, construct both timm
encoders with `pretrained=False`, load all release weights strictly, source the
trajectory vocabulary from the checkpoint, and retain checkpoint-visible
modules even when unused by the inference path.
