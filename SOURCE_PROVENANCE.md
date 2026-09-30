# Source provenance

The executable model files in this repository were copied from the research
project rather than reimplemented for GitHub.

| Public directory | Original research directory |
|---|---|
| `GLLMCDM/full_model` | `GLLMCDM_5_4_nano_attempt` |
| `GLLMCDM/without_attempt` | `GLLMCDM_5_4_nano_reasoning` |
| `baselines/gncdm_poly` | `Generative-CD-main/GNCDM_reg_me_datanew` |
| `baselines/ncdm_poly` | `NCDM` |

The post-training diagnosis scripts are also copied from the corresponding
research directories:

| Public file | Original research file |
|---|---|
| `GLLMCDM/full_model/diagnose.py` | `GLLMCDM_5_4_nano_attempt/diagnose_gllmcdm.py` |
| `GLLMCDM/without_attempt/diagnose.py` | `GLLMCDM_5_4_nano_reasoning/diagnose_gllmcdm_no_attempt_corrected.py` |
| `baselines/gncdm_poly/diagnose.py` | `Generative-CD-main/GNCDM_reg_me_datanew/diagnose_new.py` |
| `baselines/gncdm_poly/diagnose_s3.py` | `Generative-CD-main/GNCDM_reg_me_datanew/diagnose_s3_new.py` |

Only comments, module descriptions, filenames, and public command examples
were generalized. The executable diagnosis logic is unchanged.

The experiment notebooks were copied from `Exp_Incit/Exp2`,
`Exp_Incit/Exp3`, `RQ1_Paired_ZTest`, and `RQ3_TTest`. Notebook outputs are
cleared before publication to remove stale local paths and generated results.
Only path configuration and English documentation may differ from the working
copies. Model equations, prompts, training procedures, evaluation procedures,
and hyperparameters must remain unchanged.
