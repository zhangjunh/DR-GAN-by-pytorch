Celebrities in Frontal-Profile (CFP) Dataset

--------------------------------------------
Contents :
-Data
    -Images
        - #indv_id
            -frontal
               -#img_no.jpg 
            -profile
               -#img_no.jpg

    -Fiducial
        - #indv_id
            -frontal
               -#img_no.txt 
            -profile
               -#img_no.txt

    -list_names.txt

- Protocol
    -Split
        -FF
            -#split_no
                -diff.txt
                -same.txt
        -FP
            -#split_no
                -diff.txt
                -same.txt
        -Pair_list_F.txt
        -Pair_list_P.txt

-Readme.txt

-------------------------------------------
Detailed Description :
-------------------------------------------
Data : Contains images and fiducials

    Images : 10 Frontal and 4 Profile images of each 500 individuals. 
    Data/Images/xxx/frontal/xx.jpg OR Data/Images/xxx/profile/xx.jpg

    Fiducials : Frontal and Profile fiducials (30 points) of each individual. 
    Data/Images/xxx/frontal/xx.txt OR Data/Images/xxx/profile/xx.txt

    list_names.txt : Names of 500 individuals in order
--------------------------------------------
Protocol : Contains pair information for Frontal-Frontal Verification and Frontal-Profile Verification
    
    Split :
        FF : 10 fold verification for Frontal-Frontal.
        10 subfolders xx contains same.txt and diff.txt
        same.txt : 350 same pairs | diff.txt : 350 diff pairs

        FP : 10 fold verification for Frontal-Profile.
        same as FF

    Pair_list_F.txt & Pair_list_P.txt : There are 5000 Frontal and 2000 Profile images.
    In the any pair files provided in 'Split' like Split/FF/xx/same.txt,
    we provide only numbers ranging from 1-5000 for Frontal and 1-2000 for Profiles.
    The associated location of images for this number can be obtained using these .txt files.

---------------------------------------------

Training and Testing :

All 10 splits are separable in terms of identity.
We recommend performing a 10 fold cross-validation experiment : i.e.
use any 9 folds for training and 1 for testing.

To choose parameter for your algorithm : separate 9 training folds
into 1 validation and 8 training and perform 8 fold cross-validation.

---------------------------------------------

Performance Measure :

Report : Accuracy (mean and std) ; AUC (Area Under the Curve) ; EER (Equal Error Rate)

---------------------------------------------



    