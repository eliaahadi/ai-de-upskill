# table glossary
_generated 2025-11-12T21:04:10.782107Z_

## dim company
| field | type | description |
|---|---|---|
| company_id | integer | surrogate key |
| company_name | string | company name as provided in source |

## dim location
| field | type | description |
|---|---|---|
| location_id | integer | surrogate key |
| location | string | location field from source |

## dim job title
| field | type | description |
|---|---|---|
| job_title_id | integer | surrogate key |
| job_title | string | job title from source |
| experience_level | string | experience band from source |

## fact job postings
The fact table contains all original columns from the staged file plus optional surrogate keys if dimensions exist.

### columns detected in dataset
| column | inferred type | example |
|---|---|---|
| job_id | integer | 1 |
| company_name | string | Foster and Sons |
| industry | string | Healthcare |
| job_title | string | Data Analyst |
| skills_required | string | NumPy, Reinforcement Learning, PyTorch, Scikit-learn, GCP, FastAPI |
| experience_level | string | Mid |
| employment_type | string | Full-time |
| location | string | Tracybury, AR |
| salary_range_usd | string | 92860-109598 |
| posted_date | date | 2025-08-20 |
| company_size | string | Large |
| tools_preferred | string | KDB+, LangChain |

### foreign keys added during modeling
| column | references |
|---|---|
| company_id | dim_company.company_id |
| location_id | dim_location.location_id |
| job_title_id | dim_job_title.job_title_id |
