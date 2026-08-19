# Datasets and file contract

Raw datasets are downloaded from their original providers and are never
redistributed by this repository.

## UrbanNav Hong Kong

Official index and download links:

- <https://github.com/IPNL-POLYU/UrbanNavDataset>
- Medium: ROS bag, GNSS RINEX and ground truth from the official
  `UrbanNav-HK-Medium-Urban-1` section.
- Deep: ROS bags, GNSS RINEX and ground truth from the official
  `UrbanNav-HK-Deep-Urban-1` section.
- Harsh: use the provider's verified `ROS_part`, GNSS RINEX and partial ground
  truth from `UrbanNav-HK-Harsh-Urban-1`.

The paper uses three independent routes (Medium, Deep and Harsh) plus a
three-phone receiver panel on the Harsh route. The phone panel is not counted
as a fourth route.

Required inputs per route are:

1. ROS bag file(s) containing `/velodyne_points` or `/velodyne_packets`;
2. one navigation RINEX file and the observation RINEX files matched by the
   corresponding JSON in `configs/receiver_panels/`;
3. embedded `/novatel_data/inspvax` truth for Medium/Deep, or the official
   Hong Kong raw truth file passed with `--truth-csv` for Harsh;
4. the matching extrinsic JSON in `configs/extrinsics/`.

The exact physical-source mapping is in `PROVENANCE_REGISTRY.json`. In
particular, the M8T constellation partitions are dependent streams from one
receiver, not three independent receivers.

## LOCSP CR1/CR2

- DOI and public files: <https://doi.org/10.57745/YCXRWF>
- LiDAR used here: Hesai PandarXT-32 (`/hesai/pandar`)
- Estimator sources: u-blox M8T and Septentrio
- Reference only: NovAtel INSPVA

The external experiment reads both `/m8t/ublox/fix` and
`/m8t/ublox/navpvt`. It first proves that the two messages encode identical
geodetic positions, then compares:

- two physical sources without the replay stream;
- three registered streams quotienting the two M8T streams to one source;
- a deliberately naive graph treating all three streams as independent.

This LOCSP test validates source quotienting and information multiplicity. It
does not claim to validate the three-receiver triangle selector.

## Prepared-case contract

Each registered solution stream produces two files with the same stem:

- `*.estimator_input.npz`: KISS-ICP poses, GNSS timestamps, positions and
  covariances;
- `*.hidden_labels.npz`: reference trajectory used only after all candidate
  trajectories have been estimated.

The reference must never be copied into an estimator input. Extraction
reports record source bags and topics; SPP/case manifests record source paths,
parameters and hashes so the raw-to-case chain can be audited.

## Files intentionally excluded

ROS bags, RINEX files, raw point clouds, KISS-ICP caches, RTKLIB products,
prepared NPZ cases and generated trajectories are ignored by Git. They are
large derived/provider-controlled files. Keep them outside the repository or
under ignored `data/` and `outputs/` directories.
