# Archived experiment summaries

These tables are from the expanded synthetic thesis evaluation. Accuracy is aggregated by camera, noise dose, deformation or stationary acquisition, and anatomy. Nominal noise is 0 and 0.5 mm RMS. The 1 and 1.5 mm doses form the higher-noise condition.

| File | Contents |
| --- | --- |
| [summary.csv](summary.csv) | Error summaries for all nine evaluated systems and noise conditions |
| [paired_intervals.csv](paired_intervals.csv) | Paired, anatomy-stratified parent bootstrap intervals |
| [timing.csv](timing.csv) | Median and 95th-percentile processing times on the shared timing subset |

In the archived names, `pre_fit` is the learned reduced solve, `pre_refiner` is LivReNet fitted, and `livrenet` is full LivReNet. `geometry_lite` and `geometry_heavy` are standard and extended zero-initialized fitting. `no_pc` and `no_control` remove point-graph and control-graph message passing, respectively. `intermediate` adds a graph proposal refiner between the reduced solve and numerical fitting.
