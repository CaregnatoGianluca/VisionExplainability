# Model Analysis Report

This report summarizes the performance and explainability evaluation results for CNN, MAMBA, and Transformer models on CUB and CXR datasets.

## 1. Performance Evaluation

### CNN
| Metric | CUB | CXR |
| :--- | :--- | :--- |
| **Accuracy** | 0.8448 | 0.8224 |
| **Precision (Macro Avg)** | 0.8500 | 0.8221 |
| **Recall (Macro Avg)** | 0.8457 | 0.8072 |
| **F1-Score (Macro Avg)** | 0.8445 | 0.8115 |
| **AUC (Multi-class OVR)** | 0.9970 | 0.9195 |

### MAMBA
| Metric | CUB | CXR |
| :--- | :--- | :--- |
| **Accuracy** | 0.8431 | 0.6355 |
| **Precision (Macro Avg)** | 0.8522 | 0.6352 |
| **Recall (Macro Avg)** | 0.8434 | 0.6332 |
| **F1-Score (Macro Avg)** | 0.8418 | 0.6188 |
| **AUC (Multi-class OVR)** | 0.9962 | 0.8164 |

### Transformer

#### CUB Dataset
| Metric | Frozen | Unfrozen |
| :--- | :--- | :--- |
| **Accuracy** | 0.8371 | 0.8759 |
| **Precision (Macro Avg)** | 0.8409 | 0.8819 |
| **Recall (Macro Avg)** | 0.8368 | 0.8771 |
| **F1-Score (Macro Avg)** | 0.8358 | 0.8746 |
| **AUC (Multi-class OVR)** | 0.9976 | 0.9976 |

#### CXR Dataset
| Metric | Frozen | Unfrozen |
| :--- | :--- | :--- |
| **Accuracy** | 0.7383 | 0.8131 |
| **Precision (Macro Avg)** | 0.7276 | 0.8116 |
| **Recall (Macro Avg)** | 0.7168 | 0.8209 |
| **F1-Score (Macro Avg)** | 0.7108 | 0.8127 |
| **AUC (Multi-class OVR)** | 0.8331 | 0.8989 |

---

## 2. Explainability Results
*Note: The first table represents the Mean values, and the second table represents the Standard Deviation.*

### CNN

#### CUB
**Mean**
| Method | Train | JSS | Chi2 | PCC |
| :--- | :--- | :--- | :--- | :--- |
| **AblationCAM** | False | 0.604966 | 0.457849 | 0.740222 |
| **AblationCAM** | True | 0.600456 | 0.447721 | 0.741263 |
| **GradCAM** | False | 0.612339 | 0.473211 | 0.736168 |
| **GradCAM** | True | 0.606742 | 0.461756 | 0.738568 |
| **ScoreCAM** | False | 0.571234 | 0.367452 | 0.693527 |
| **ScoreCAM** | True | 0.561988 | 0.341270 | 0.682213 |

**Standard Deviation**
| Method | Train | JSS | Chi2 | PCC |
| :--- | :--- | :--- | :--- | :--- |
| **AblationCAM** | False | 0.080788 | 0.208776 | 0.140145 |
| **AblationCAM** | True | 0.079446 | 0.204698 | 0.135525 |
| **GradCAM** | False | 0.084124 | 0.215047 | 0.146940 |
| **GradCAM** | True | 0.080555 | 0.205313 | 0.138556 |
| **ScoreCAM** | False | 0.087347 | 0.237501 | 0.146164 |
| **ScoreCAM** | True | 0.089945 | 0.247093 | 0.145308 |

#### CXR
**Mean**
| Method | Train | JSS | Chi2 | PCC |
| :--- | :--- | :--- | :--- | :--- |
| **AblationCAM** | False | 0.638251 | 0.514056 | 0.214263 |
| **AblationCAM** | True | 0.649943 | 0.541944 | 0.251579 |
| **GradCAM** | False | 0.545037 | 0.265068 | 0.152263 |
| **GradCAM** | True | 0.569470 | 0.333981 | 0.197630 |
| **ScoreCAM** | False | 0.736467 | 0.734409 | 0.352235 |
| **ScoreCAM** | True | 0.743222 | 0.747266 | 0.370415 |

**Standard Deviation**
| Method | Train | JSS | Chi2 | PCC |
| :--- | :--- | :--- | :--- | :--- |
| **AblationCAM** | False | 0.110912 | 0.245945 | 0.307349 |
| **AblationCAM** | True | 0.111432 | 0.247074 | 0.279211 |
| **GradCAM** | False | 0.133924 | 0.361472 | 0.280955 |
| **GradCAM** | True | 0.131008 | 0.344498 | 0.264032 |
| **ScoreCAM** | False | 0.038492 | 0.075140 | 0.192921 |
| **ScoreCAM** | True | 0.037782 | 0.072049 | 0.188656 |

---

### MAMBA

#### CUB
**Mean**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.341008 | -0.362662 | 0.131520 |
True | 0.342304 | -0.357892 | 0.135005 |

**Standard Deviation**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.043008 | 0.157956 | 0.271943 |
True | 0.043194 | 0.158329 | 0.264175 |

#### CXR
**Mean**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.821414 | 0.874262 | -0.010853 |
True | 0.738452 | 0.733930 | -0.137167 |

**Standard Deviation**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.017119 | 0.023483 | 0.079180 |
True | 0.040565 | 0.076871 | 0.156199 |

---

### Transformer

#### CUB (Frozen)
**Mean**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.523938 | 0.243472 | 0.566925 |
True | 0.522406 | 0.238594 | 0.562504 |

**Standard Deviation**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.063824 | 0.190785 | 0.153774 |
True | 0.065380 | 0.195577 | 0.155712 |

#### CUB (Unfrozen)
**Mean**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.539171 | 0.286727 | 0.604711 |
True | 0.534215 | 0.272464 | 0.595921 |

**Standard Deviation**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.067026 | 0.195800 | 0.152661 |
True | 0.067889 | 0.200118 | 0.156581 |

#### CXR (Frozen)
**Mean**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.652867 | 0.558339 | -0.043306 |
True | 0.657902 | 0.570228 | -0.016654 |

**Standard Deviation**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.036798 | 0.087856 | 0.117438 |
True | 0.037048 | 0.087318 | 0.127296 |

#### CXR (Unfrozen)
**Mean**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.482235 | 0.084622 | 0.156432 |
True | 0.482120 | 0.084229 | 0.161501 |

**Standard Deviation**
Train | JSS | Chi2 | PCC |
:--- | :--- | :--- | :--- |
False | 0.056015 | 0.176778 | 0.134989 |
True | 0.058065 | 0.182995 | 0.143574 |
