declare -a revision_numbers=("2908" "2913" "2935" "3016" "3057" "3099" "3128" "3141" "3152" "3184" "3152" "3184" "3231" "3249" "3298" "3328" "3337" "3408" "3412" "3413" "3553" "3577" "3598" "3670" "3679" "3695" "3715" "3747" "3760" "3776" "3815" "3838" "3874" "3923" "3953" "4040" "4050" "4168" "4169" "4170" "4197" "4255" "4256" "4292" 
"4333" "4334" "4336" "4540" "4548" "4769" "5027" "5236" "5316" "5321" "5686" "5709" "5711" "5892" "6097")
for revision_no in "${revision_numbers[@]}"
do
    wget -O "./data/R-data/R-policies/CRAN/policies.r${revision_no}.html" "https://raw.githubusercontent.com/eddelbuettel/crp/master/html/policies.r${revision_no}.html"
    wget -O "./data/R-data/R-policies/CRAN/policies.r${revision_no}.txt" "https://raw.githubusercontent.com/eddelbuettel/crp/master/txt/policies.r${revision_no}.txt"
done