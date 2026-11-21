# Mech Royale — Survival & Tactical Map-Aware Bot

A standard-library Python bot for the Builders Buildathon **Mech Royale** track (`mechfight/1`).

## Features

- Loads `/arena/v2/world.json` and `/arena/v2/terrain.i16` to build a 2D navigation grid (wall clearance + cover score) and fast 3D prism/terrain line-of-sight raycaster.
- Survival-first state machine: zone safety, cover retreat & out-of-combat regen, rear-armor orientation, projectile dodging, and supply drop/cube opportunism.
- Exact quadratic projectile lead (`gun`), ballistic solver (`mortar`), charge-and-release (`laser`), and close-range counter (`lightning`).

## Open-Source Dependencies

- Python 3.13 standard library only (plus `mechfight.py` from the official `mechfight-kit`).
