# Init Communicator

We previously observed unexpected long init times (3+ minutes) for FSDP/PP communicators on the h200 server (coriander), with timing variations across different GPU combinations.

This directory contains tests for p2p/collective operations using both torch.dist only and ray actor (with cupy and torch.dist communicators) under standard Ray dependency.

Key findings:
- Initialization now takes between 3-40 seconds for both cupy/torch.dist communicators in ray.
- Performance still varies by GPU pairing (fastest: pairs 1,2 and 2,3).

Key examples:
- [Ray actor collective with cupy or torch.dist](./actor/coll/compiled/example.py).
- [Ray actor p2p with cupy](./actor/p2p/compiled/cupy/example.py).
- [Ray actor p2p with torch.dist](./actor/p2p/compiled/distributed/example.py).

These results suggest our FSDP/PP implementation branches may have introduced communicator initialization issues. Next, we'll benchmark FSDP/PP communicator initialization with standard Ray dependency.
