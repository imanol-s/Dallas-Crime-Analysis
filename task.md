# V2 Roadmap Task Log

Master execution ledger for
[`v2_implementation_roadmap.md`](/Users/imanol/Documents/Projects/Dallas-Crime-Analysis/v2_implementation_roadmap.md).
Read this at session start with
[`AGENTS.md`](/Users/imanol/Documents/Projects/Dallas-Crime-Analysis/AGENTS.md) and
[`RUNLOG.md`](/Users/imanol/Documents/Projects/Dallas-Crime-Analysis/RUNLOG.md).

## Legend

- Status: `completed`, `in_progress`, `ready`, `blocked`, `out_of_scope`
- Scope: `repo_native`, `external_dependency`, `non_package`
- Role mapping:
  - `project-manager`: tracker state, dependencies, checkpoints, blocked rationale
  - `data-engineer`: `acquire`/`build` and roadmap analytics-engineer work
  - `data-scientist`: `analyze`, modeling, validation, forecasting, benchmark/report logic

## Definition Of Done Status

- `repo_native` roadmap rows: complete
- `external_dependency` / `non_package` rows: explicitly tracked below and excluded from package DoD

## Q1

### Week 1-2: Project Kickoff & Data Planning

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1-W1-1 | Secure stakeholder approval and budget allocation | project-manager | blocked | external_dependency | External sponsor decision | Roadmap-only item; no in-package approval artifact | Track unblocker and decision owner in this log |
| Q1-W1-2 | Recruit/assign core team members | project-manager | blocked | external_dependency | Stakeholder approval | Role skills and agent notes exist, but no staffing artifact | Track assignment owner and kickoff prerequisite |
| Q1-W1-3 | Document data collection requirements | project-manager + data-engineer + data-scientist | completed | repo_native | None | `README.md`, `docs/methodology.md`, `docs/source_dictionary.md`, `docs/refresh_workflow.md` | Keep docs aligned with new artifacts |
| Q1-W1-4 | Identify external data sources and access methods | data-engineer | completed | repo_native | Q1-W1-3 | `v2_implementation_roadmap.md`, `V2_EXECUTION_TRACKER.md`, housing/acquisition docs, source notes in `RUNLOG.md` | Extend to sidecar-specific source notes as categories land |

### Week 3-5: Historical Crime & Housing Data Collection

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1-W3-1 | Collect 3-year historical crime data | data-engineer | completed | repo_native | Q1-W1-3 | `crime_history_records.csv` support in `src/dallas_crime/acquire/crime.py`; panel consumption in `build.py`; acquire tests | Re-validate against live refresh when running full `acquire` |
| Q1-W3-2 | Retrieve historical housing price data | data-engineer | completed | repo_native | Q1-W1-3 | `housing_market_history.csv`, `housing_history_panel.csv`, `housing_history_features.csv` support and tests | Maintain coverage metrics during refreshes |
| Q1-W3-3 | Gather demographic snapshots (Census, ACS) | data-engineer | completed | repo_native | Q1-W3-1 | `acs_zcta_snapshots.csv` support in `src/dallas_crime/acquire/census.py`; acquire tests | Promote snapshots into processed features |
| Q1-W3-4 | Build time series database | data-engineer | out_of_scope | non_package | Q1-W3-1, Q1-W3-2, Q1-W3-3 | Current package is file-based CLI; `V2_EXECUTION_TRACKER.md` marks DB/service work out of package | Track separately if repo boundary expands |
| Q1-W3-5 | Data validation and quality assessment | data-engineer + data-scientist | completed | repo_native | Q1-W3-1, Q1-W3-2, Q1-W3-3 | QA CSV/JSON artifacts in `build.py`, `source_completeness_scores.csv`, expanded build/analyze tests, and smoke coverage | Re-validate against fresh acquired data as new sidecars land |

### Week 6-8: Exploratory Analysis & Clustering

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1-W6-1 | Perform univariate & temporal trend analysis | data-scientist | completed | repo_native | Q1-W3-1 | `crime_trend_decomposition.csv`, `crime_trend_decomposition.md`, updated summary report, and analyze tests | Re-run on live refreshed history to confirm decomposition stability |
| Q1-W6-2 | Conduct clustering analysis | data-scientist | completed | repo_native | Q1-W3-1, Q1-W3-3 | `cluster_assignments.csv`, `cluster_profiles.csv`, `zip_benchmarks.csv`, `benchmark_summary.md`, and analyze tests | Revisit cluster feature inputs after additional interaction features land |
| Q1-W6-3 | Perform spatial analysis (Moran's I, hot spots) | data-scientist | completed | repo_native | Q1-W6-2 | `spatial_diagnostics.csv`, `spatial_hotspots.csv`, and tests | Re-validate on live refresh as data evolves |
| Q1-W6-4 | Create comprehensive segmentation visualizations | data-scientist | completed | repo_native | Q1-W6-2, Q1-W6-3 | `docs/dashboard_specifications.md`, benchmark summary outputs, and trend/forecast artifact inventory | Convert specs into an external dashboard implementation only if package scope expands |

## Q2

### Week 1-2: External Data Integration

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q2-W1-1 | Economic indicators integration | data-engineer | completed | repo_native | Q1-W3-5 | `acquire` now materializes `dfw_zip_economic_sidecar.csv` with ZIP-level economic index, median wage, unemployment, and trend fields; additive build join validated in tests/smoke | Maintain refresh cadence via scheduled `acquire` runs |
| Q2-W1-2 | Real estate market data beyond current feeds | data-engineer | completed | repo_native | Q1-W3-2 | `acquire` now materializes `dfw_zip_real_estate_sidecar.csv` with investor-share and pressure metrics from housing + ACS context; additive join validated | Refit formulas only if source contracts change |
| Q2-W1-3 | Law enforcement data integration | data-engineer | completed | repo_native | Q1-W3-5 | `acquire` now materializes `dfw_zip_law_enforcement_sidecar.csv` with staffing/arrest/violent-rate features (including local arrests feed when available) | Monitor refresh stability as arrests source updates |
| Q2-W1-4 | Social services data integration | data-engineer | completed | repo_native | Q1-W3-3 | `acquire` now materializes `dfw_zip_social_services_sidecar.csv` with educational-attainment and assistance-share features plus access score | Keep metric definitions aligned with policy interpretation |
| Q2-W1-5 | Infrastructure & environment integration | data-engineer | completed | repo_native | Q1-W3-5 | `acquire` now materializes `dfw_zip_infrastructure_sidecar.csv` with transit/vacancy context and infrastructure scoring | Revalidate if transit/vacancy source fields change |

### Week 3-5: Feature Engineering & Selection

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q2-W3-1 | Create temporal features | data-engineer | completed | repo_native | Q1-W3-1, Q1-W3-3 | Expanded `crime_history_features.csv`, `acs_snapshot_features.csv`, additive joins into `model_dataset.csv`, and pipeline tests | Revalidate temporal feature stability after each live acquire refresh |
| Q2-W3-2 | Engineer interaction & aggregate features | data-engineer | completed | repo_native | Q2-W1-1 to Q2-W1-5 | `interaction_features.csv` now emits 12 additive interaction/aggregate terms and `build_all()` left-joins them into `model_dataset.csv`; covered in `tests/test_pipeline.py` and smoke | Keep interaction set additive as new sidecar categories gain live coverage |
| Q2-W3-3 | Derive calculated features | data-engineer | completed | repo_native | Q2-W3-1 | Crime momentum/acceleration fields, ACS snapshot change/slope fields, and `source_completeness_overall_score` now land in processed outputs | Add interaction-style derived features after external sidecars are populated with live data |
| Q2-W3-4 | Perform feature selection | data-scientist | completed | repo_native | Q2-W3-1, Q2-W3-2, Q2-W3-3 | `feature_selection_metrics.csv` and `feature_selection_notes.md` now ship with candidate ranking, availability, univariate diagnostics, and recommendation flags; covered by `tests/test_project.py` | Recalibrate selection thresholds as additional external signals are populated |

### Week 6-10: Model Development & Validation

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q2-W6-1 | Train baseline models | data-scientist | completed | repo_native | Q2-W3-4 | HC3 baseline/expanded regressions plus predictive-family baseline tier now emit in `predictive_model_metrics.csv` and `predictive_model_predictions.csv` with validation metrics | Re-benchmark baseline error after each full data refresh |
| Q2-W6-2 | Train intermediate models | data-scientist | completed | repo_native | Q2-W6-1 | Intermediate predictive tier (`intermediate_market_augmented`) and forecast-family comparisons now emit in `predictive_model_metrics.csv` and `forecast_model_metrics.csv` | Add additional intermediate candidates only if they improve out-of-sample metrics |
| Q2-W6-3 | Train advanced models | data-scientist | completed | repo_native | Q2-W6-2 | Advanced predictive tier (`advanced_interaction_model`) plus trend/forecast artifacts (`crime_trend_decomposition.csv`, `crime_forecasts.csv`) are implemented and tested | Keep advanced tier deterministic and dependency-light unless scope expands |
| Q2-W6-4 | Train specialized models | data-scientist | completed | repo_native | Q2-W6-2 | Specialized tier (`specialized_segment_model`) using segment context now emits in predictive artifacts and is validated in tests | Refit specialized tier after major cluster-definition updates |
| Q2-W6-5 | Model selection & ensembling | data-scientist | completed | repo_native | Q2-W6-1, Q2-W6-2, Q2-W6-3, Q2-W6-4 | Model ranking/selection is captured in `predictive_model_metrics.csv` and `model_selection_notes.md`; weighted ensemble row `ensemble_top2_inverse_rmse` now emits in predictive outputs | Monitor uplift of ensemble vs best single model during refreshes |

### Week 11: Forecasting Setup

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q2-W11-1 | Implement 12-month forecast models | data-scientist | completed | repo_native | Q2-W6-5 | `crime_forecasts.csv`, `forecast_model_metrics.csv`, `forecast_notes.md`, and analyze tests | Reassess selected models after live multi-year refreshes |
| Q2-W11-2 | Build scenario analysis framework | data-scientist | completed | repo_native | Q2-W11-1 | `scenario_impacts.csv` and `scenario_notes.md` now ship with deterministic planning scenarios | Expand beyond rule-based multipliers if a causal framework is later justified |
| Q2-W11-3 | Create forecast confidence intervals | data-scientist | completed | repo_native | Q2-W11-1 | `forecast_confidence_intervals.csv` now emits 80% and 95% interval bounds per ZIP/horizon | Recalibrate interval width once longer histories are refreshed |

## Q3

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q3-W1-2 | Comprehensive validation framework | data-scientist | completed | repo_native | Q2-W6-5 | `comprehensive_validation_metrics.csv` and `comprehensive_validation_notes.md` now aggregate regression, predictive-family, forecast, drift, and scenario checks; tested in `tests/test_project.py` | Re-run validation rollup after major feature/model revisions |
| Q3-W1-2-B | Sensitivity analysis | data-scientist | completed | repo_native | Q2-W11-2 | `scenario_impacts.csv` and `scenario_notes.md` now provide repeatable what-if sensitivity views | Expand sensitivity dimensions if additional causal assumptions are introduced later |
| Q3-W3-4 | Stress testing (5+ scenarios) | data-scientist | completed | repo_native | Q2-W11-2 | `scenario_impacts.csv` now emits five deterministic scenarios (`baseline`, `economic_shock`, `housing_correction`, `policing_expansion`, `systemic_shock`) with notes and tests | Expand scenario library only when stakeholders request additional policy levers |
| Q3-W3-4-B | Drift detection system | data-scientist | completed | repo_native | Q3-W1-2 | `model_drift_diagnostics.csv`, `model_drift_notes.md`, analyze tests, and smoke coverage | Extend beyond crime-rate drift if housing-side drift becomes a package requirement |
| Q3-W5-6 | Policy recommendations by segment | data-scientist + project-manager | completed | repo_native | Q3-W3-4, Q3-W3-4-B | `policy_recommendations_by_segment.csv` and `policy_recommendations_by_segment.md` now synthesize benchmark + scenario outputs into segment-priority actions; covered in tests | Refresh recommendations after each scenario-parameter update |
| Q3-W5-6-B | Benchmarking framework | data-scientist | completed | repo_native | Q1-W6-2, Q3-W1-2 | `zip_benchmarks.csv`, `benchmark_summary.md`, cluster merges, and analyze tests now exist | Add policy-facing synthesis only if downstream consumers need it inside this repo |
| Q3-W7-9 | Dashboard design & specifications | project-manager + data-scientist | completed | repo_native | Q3-W5-6, Q3-W5-6-B | `docs/dashboard_specifications.md` now maps five dashboard surfaces to concrete package artifacts | Keep the spec aligned if the artifact contract changes again |

## Q4

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q4-W1-2 | Backend development (data APIs, refresh logic) | project-manager | out_of_scope | non_package | Q3-W7-9 | Current package is CLI/file-artifact based | Track only if service repo/app is added |
| Q4-W3-4 | Frontend development (UI, interactivity) | project-manager | out_of_scope | non_package | Q4-W1-2 | No frontend app in current package | Track only as future non-package work |
| Q4-W5-6 | Integration testing, UAT, performance optimization | project-manager | out_of_scope | non_package | Q4-W1-2, Q4-W3-4 | No deployable dashboard/application | Track in external delivery plan if app exists |
| Q4-W7-8 | User training, documentation, go-live support | project-manager | out_of_scope | non_package | Q4-W5-6 | No live operational system in current package | Track externally with stakeholder plan |

## Cross-Cutting Workstreams

### Documentation (Ongoing)

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DOC-W1-2 | Create project wiki / architecture decision records | project-manager | completed | repo_native | None | ADR-style and index docs now exist in `docs/architecture_decisions.md` and `docs/technical_documentation_index.md` | Keep ADR/index updated as architecture boundaries change |
| DOC-W3+ | Daily standup notes | project-manager | blocked | external_dependency | Team staffing | Requires active delivery team process | Track outside repo-native implementation |
| DOC-W3+-B | Methodology documentation | project-manager + data-scientist | completed | repo_native | Ongoing feature work | `README.md`, `docs/methodology.md`, `docs/dashboard_specifications.md`, and `V2_EXECUTION_TRACKER.md` are now synchronized to the expanded V2 artifact contract | Continue same-day doc updates for future additive artifacts |
| DOC-FINAL | Consolidate 100+ page technical documentation | project-manager | completed | repo_native | Most roadmap tranches | Consolidated package compendium now exists in `docs/v2_technical_compendium.md` and is indexed by `docs/technical_documentation_index.md` | Keep compendium synchronized as future roadmap increments land |

### Quality Assurance (Ongoing)

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QA-1 | Data validation tests (automated) | data-engineer | completed | repo_native | Q1-W3-5 | `tests/test_pipeline.py`, `tests/test_project.py`, full `pytest -q`, and `make smoke` now cover snapshot/sidecar/forecast/drift paths | Keep fixtures aligned with future artifact additions |
| QA-2 | Model performance monitoring (weekly) | project-manager + data-scientist | blocked | external_dependency | Repeated production runs | Current repo supports offline diagnostics, not weekly operations | Track as external operating cadence |
| QA-3 | Code reviews (all commits) | project-manager | blocked | external_dependency | Team process | Requires human review workflow | Track as governance item, not code task |
| QA-4 | Monthly quality assessment | project-manager + data-engineer | blocked | external_dependency | Scheduled recurring execution | No always-on scheduler in current package | Track until automation/service boundary exists |

### Stakeholder Engagement (Ongoing)

| Roadmap Ref | Task | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STAKE-1 | Bi-weekly project status meetings | project-manager | blocked | external_dependency | Stakeholder approval | Organizational process only | Track meeting owner/unblocker |
| STAKE-2 | Monthly demos of progress | project-manager | blocked | external_dependency | STAKE-1 | No recurring stakeholder cadence in repo | Track externally |
| STAKE-3 | Quarterly steering committee reviews | project-manager | blocked | external_dependency | STAKE-1 | Organizational process only | Track externally |
| STAKE-4 | User feedback collection (UAT phase) | project-manager | blocked | external_dependency | Q4-W5-6 | No UAT system/app in current package | Track with future app rollout |

## Decision Checkpoints

| Checkpoint | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| End Q1 Go/No-Go: historical data complete and validated | project-manager | completed | repo_native | Q1-W3-5 | Historical crime, housing, ACS snapshots, completeness scoring, expanded tests, and smoke verification are all in place | Recheck only after the next live acquire refresh materially changes coverage |
| End Q1 Go/No-Go: clustering insights actionable | project-manager + data-scientist | completed | repo_native | Q1-W6-2 | Segmentation, benchmark, drift, and dashboard-spec artifacts now provide actionable cluster framing inside the package | External stakeholders still need separate signoff outside the repo |
| End Q2 Go/No-Go: model R² > 0.45 | project-manager + data-scientist | completed | repo_native | Q2-W6-5 | Baseline/expanded regression metrics and predictive-family comparisons are now implemented and repeatedly validated above threshold in package runs | Reconfirm on each live data refresh before stakeholder review |
| End Q2 Go/No-Go: feature engineering captured 90%+ power | project-manager + data-scientist | completed | repo_native | Q2-W3-4 | `feature_power_retention_metrics.csv` and `feature_power_retention_notes.md` now emit explicit retention-ratio thresholds and checkpoint pass/fail state; covered in `tests/test_project.py` and smoke | Recompute checkpoint status each time feature candidates or model-family selection logic changes |
| End Q3 Go/No-Go: validation & drift detection complete | project-manager + data-scientist | completed | repo_native | Q3-W1-2, Q3-W3-4-B | Comprehensive validation rollup, 5-scenario stress testing, and drift diagnostics now ship with tests and smoke coverage | Re-run checkpoint after any model-family changes |
| End Q3 Go/No-Go: dashboard requirements approved | project-manager | blocked | external_dependency | Q3-W7-9 | No stakeholder approval artifact | Track external signoff after dashboard specs exist |

## Risks

| Risk | Owner | Status | Scope | Depends On | Evidence | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| Data access delays | project-manager + data-engineer | completed | repo_native | Historical sidecar gap | Sidecar generation is now built into `acquire` and all five category files (`economic`, `real_estate`, `law_enforcement`, `social_services`, `infrastructure`) are populated and flowing into `model_dataset.csv` with category completeness at `1.0` for modeled ZIPs | Monitor source freshness as routine operations, not as an open delivery blocker |
| Model underperforms | data-scientist | completed | repo_native | Q2 predictive tranche | Predictive-family metrics, ensemble selection diagnostics, drift diagnostics, and feature-power retention checkpoint artifacts now provide explicit performance guardrails | Maintain recurring performance checks through the existing report artifacts |
| Dashboard scope creep | project-manager | out_of_scope | non_package | Q3/Q4 app work | No dashboard app in current package | Contain to dashboard specs only |
| Key personnel unavailable | project-manager | blocked | external_dependency | Staffing | External staffing decision required | Track externally |
| Budget overruns | project-manager | blocked | external_dependency | Stakeholder approval | External budget decision required | Track externally |
