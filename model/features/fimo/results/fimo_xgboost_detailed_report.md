# FIMO XGBOOST Feature Analysis Report

## Summary of File Analysis
- Total files found and processed: 70
- Features analyzed: [3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

## Accuracy Table
| Feature Count | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |
|---------------|--------|--------|--------|--------|--------|--------|
| 3 | 0.5986 | 0.6203 | 0.5550 | 0.6101 | 0.5692 | 0.5906 |
| 10 | 0.7606 | 0.6879 | 0.6870 | 0.6858 | 0.6897 | 0.7022 |
| 20 | 0.7770 | 0.7256 | 0.7213 | 0.7844 | 0.8128 | 0.7642 |
| 30 | 0.8192 | 0.7773 | 0.7555 | 0.8303 | 0.8308 | 0.8026 |
| 40 | 0.8380 | 0.8091 | 0.7457 | 0.8463 | 0.8462 | 0.8171 |
| 50 | 0.8545 | 0.8270 | 0.7775 | 0.8876 | 0.8333 | 0.8360 |
| 60 | 0.8075 | 0.7674 | 0.7971 | 0.8693 | 0.8667 | 0.8216 |
| 70 | 0.8404 | 0.7972 | 0.7848 | 0.8899 | 0.8538 | 0.8332 |
| 80 | 0.8521 | 0.8032 | 0.7824 | 0.8899 | 0.8795 | 0.8414 |
| 90 | 0.8521 | 0.8151 | 0.7775 | 0.8761 | 0.8769 | 0.8396 |
| 100 | 0.8756 | 0.8191 | 0.7873 | 0.8899 | 0.8949 | 0.8533 |
| 110 | 0.8779 | 0.8111 | 0.7751 | 0.8991 | 0.8897 | 0.8506 |
| 120 | 0.8756 | 0.8191 | 0.7848 | 0.9037 | 0.8795 | 0.8525 |
| 130 | 0.8826 | 0.8250 | 0.8020 | 0.8830 | 0.8718 | 0.8529 |

## F1 Score Table
| Feature Count | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |
|---------------|--------|--------|--------|--------|--------|--------|
| 3 | 0.6376 | 0.6433 | 0.5843 | 0.6444 | 0.5913 | 0.6202 |
| 10 | 0.7672 | 0.7068 | 0.6987 | 0.6879 | 0.7021 | 0.7125 |
| 20 | 0.7827 | 0.7430 | 0.7287 | 0.7802 | 0.8104 | 0.7690 |
| 30 | 0.8268 | 0.7804 | 0.7546 | 0.8278 | 0.8289 | 0.8037 |
| 40 | 0.8400 | 0.8136 | 0.7405 | 0.8395 | 0.8438 | 0.8155 |
| 50 | 0.8579 | 0.8308 | 0.7762 | 0.8839 | 0.8305 | 0.8359 |
| 60 | 0.8055 | 0.7781 | 0.7951 | 0.8696 | 0.8671 | 0.8231 |
| 70 | 0.8395 | 0.8037 | 0.7864 | 0.8889 | 0.8519 | 0.8341 |
| 80 | 0.8509 | 0.8080 | 0.7828 | 0.8899 | 0.8779 | 0.8419 |
| 90 | 0.8525 | 0.8197 | 0.7810 | 0.8782 | 0.8760 | 0.8415 |
| 100 | 0.8759 | 0.8236 | 0.7885 | 0.8909 | 0.8935 | 0.8545 |
| 110 | 0.8766 | 0.8169 | 0.7775 | 0.9008 | 0.8895 | 0.8523 |
| 120 | 0.8746 | 0.8236 | 0.7864 | 0.9053 | 0.8788 | 0.8537 |
| 130 | 0.8793 | 0.8306 | 0.8000 | 0.8852 | 0.8704 | 0.8531 |

## Precision Table
| Feature Count | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |
|---------------|--------|--------|--------|--------|--------|--------|
| 3 | 0.6952 | 0.6737 | 0.6287 | 0.6991 | 0.6242 | 0.6642 |
| 10 | 0.7751 | 0.7353 | 0.7147 | 0.6900 | 0.7215 | 0.7273 |
| 20 | 0.7895 | 0.7726 | 0.7382 | 0.7766 | 0.8086 | 0.7771 |
| 30 | 0.8379 | 0.7839 | 0.7537 | 0.8258 | 0.8275 | 0.8057 |
| 40 | 0.8423 | 0.8194 | 0.7362 | 0.8362 | 0.8424 | 0.8153 |
| 50 | 0.8625 | 0.8358 | 0.7750 | 0.8826 | 0.8287 | 0.8369 |
| 60 | 0.8036 | 0.7945 | 0.7933 | 0.8699 | 0.8676 | 0.8258 |
| 70 | 0.8387 | 0.8128 | 0.7882 | 0.8881 | 0.8507 | 0.8357 |
| 80 | 0.8499 | 0.8145 | 0.7832 | 0.8899 | 0.8771 | 0.8429 |
| 90 | 0.8529 | 0.8259 | 0.7854 | 0.8811 | 0.8754 | 0.8441 |
| 100 | 0.8762 | 0.8297 | 0.7897 | 0.8920 | 0.8929 | 0.8561 |
| 110 | 0.8755 | 0.8252 | 0.7803 | 0.9033 | 0.8894 | 0.8547 |
| 120 | 0.8737 | 0.8297 | 0.7882 | 0.9077 | 0.8783 | 0.8555 |
| 130 | 0.8775 | 0.8391 | 0.7983 | 0.8884 | 0.8695 | 0.8546 |

## Recall Table
| Feature Count | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |
|---------------|--------|--------|--------|--------|--------|--------|
| 3 | 0.5986 | 0.6203 | 0.5550 | 0.6101 | 0.5692 | 0.5906 |
| 10 | 0.7606 | 0.6879 | 0.6870 | 0.6858 | 0.6897 | 0.7022 |
| 20 | 0.7770 | 0.7256 | 0.7213 | 0.7844 | 0.8128 | 0.7642 |
| 30 | 0.8192 | 0.7773 | 0.7555 | 0.8303 | 0.8308 | 0.8026 |
| 40 | 0.8380 | 0.8091 | 0.7457 | 0.8463 | 0.8462 | 0.8171 |
| 50 | 0.8545 | 0.8270 | 0.7775 | 0.8876 | 0.8333 | 0.8360 |
| 60 | 0.8075 | 0.7674 | 0.7971 | 0.8693 | 0.8667 | 0.8216 |
| 70 | 0.8404 | 0.7972 | 0.7848 | 0.8899 | 0.8538 | 0.8332 |
| 80 | 0.8521 | 0.8032 | 0.7824 | 0.8899 | 0.8795 | 0.8414 |
| 90 | 0.8521 | 0.8151 | 0.7775 | 0.8761 | 0.8769 | 0.8396 |
| 100 | 0.8756 | 0.8191 | 0.7873 | 0.8899 | 0.8949 | 0.8533 |
| 110 | 0.8779 | 0.8111 | 0.7751 | 0.8991 | 0.8897 | 0.8506 |
| 120 | 0.8756 | 0.8191 | 0.7848 | 0.9037 | 0.8795 | 0.8525 |
| 130 | 0.8826 | 0.8250 | 0.8020 | 0.8830 | 0.8718 | 0.8529 |

## Auc Table
| Feature Count | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |
|---------------|--------|--------|--------|--------|--------|--------|
| 3 | 0.5003 | 0.5055 | 0.4440 | 0.5536 | 0.4524 | 0.4912 |
| 10 | 0.7341 | 0.6622 | 0.6197 | 0.6210 | 0.7105 | 0.6695 |
| 20 | 0.7581 | 0.7252 | 0.6721 | 0.7283 | 0.8175 | 0.7402 |
| 30 | 0.8702 | 0.7753 | 0.7503 | 0.8587 | 0.8645 | 0.8238 |
| 40 | 0.8565 | 0.8093 | 0.7904 | 0.8580 | 0.8770 | 0.8382 |
| 50 | 0.8742 | 0.8120 | 0.7559 | 0.8814 | 0.9054 | 0.8458 |
| 60 | 0.8516 | 0.7790 | 0.8240 | 0.8670 | 0.9220 | 0.8487 |
| 70 | 0.8732 | 0.8065 | 0.8539 | 0.8837 | 0.9242 | 0.8683 |
| 80 | 0.8924 | 0.8073 | 0.8397 | 0.8849 | 0.9403 | 0.8729 |
| 90 | 0.8951 | 0.8128 | 0.8434 | 0.8926 | 0.9408 | 0.8769 |
| 100 | 0.9017 | 0.8045 | 0.8537 | 0.9045 | 0.9501 | 0.8829 |
| 110 | 0.9065 | 0.8095 | 0.8591 | 0.8914 | 0.9492 | 0.8831 |
| 120 | 0.9120 | 0.8185 | 0.8649 | 0.8964 | 0.9465 | 0.8877 |
| 130 | 0.9084 | 0.8198 | 0.8654 | 0.8777 | 0.9470 | 0.8837 |

## Conclusion

This report provides a detailed analysis of FIMO XGBOOST model performance across different feature counts and cross-validation folds.

### Best Performing Feature Counts

- Best Accuracy: 0.8533 (with 100 features)
- Best F1 Score: 0.8545 (with 100 features)
- Best Precision: 0.8561 (with 100 features)
- Best Recall: 0.8533 (with 100 features)
- Best Auc: 0.8877 (with 120 features)
