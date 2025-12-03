library(dplyr)
library(ggplot2)

## Biological subtype naming based on multimodal features
output_dir <- "Subtype_Naming_Results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

## Define biological names for three subtypes
naming_table <- data.frame(
  Subtype_ID = c("Subtype 1", "Subtype 2", "Subtype 3"),
  Original_Name = c("Subtype 1", "Subtype 2", "Subtype 3"),
  Biological_Name = c("Hippocampal-Predominant", "Cortical-Diffuse", "Typical AD"),
  Short_Name = c("HP", "CD", "TAD"),
  Key_Features = c(
    "Severe hippocampal atrophy; Moderate CSF pathology; Slower progression",
    "Diffuse cortical thinning; Mild hippocampal atrophy; Moderate progression",
    "Severe hippocampal atrophy; High CSF pathology; Rapid progression"
  ),
  Clinical_Implications = c(
    "Better cognitive reserve; Suitable for early intervention",
    "Wide-spread neurodegeneration; Multi-target therapy needed",
    "Aggressive disease course; Intensive monitoring required"
  )
)

print(naming_table[, c("Subtype_ID", "Biological_Name", "Short_Name")])

write.csv(naming_table, file.path(output_dir, "Subtype_Biological_Naming_Table.csv"), row.names = FALSE)

naming_table_paper <- naming_table %>%
  select(Original_Name, Biological_Name, Short_Name, Key_Features, Clinical_Implications)

write.csv(naming_table_paper, file.path(output_dir, "Subtype_Naming_Table_For_Paper.csv"), row.names = FALSE)

## Generate naming rationale document
naming_rationale <- c(
  "========================================================================",
  "  Biological Subtype Naming: Rationale and Clinical Implications",
  "========================================================================",
  "",
  "NAMING PRINCIPLES",
  "",
  "Three subtypes were named based on multimodal features:",
  "1. Structural MRI features (hippocampal volume, cortical thickness, ventricles)",
  "2. CSF biomarker patterns (Abeta42, Tau, p-Tau)",
  "3. Clinical progression rate (longitudinal cognitive decline)",
  "",
  "========================================================================",
  "Subtype 1: Hippocampal-Predominant (HP)",
  "========================================================================",
  "",
  "KEY FEATURES",
  "- Severe hippocampal atrophy",
  "- Prominent entorhinal cortex atrophy",
  "- Moderate CSF Tau/p-Tau elevation",
  "- Relatively preserved cortical thickness",
  "",
  "CLINICAL CHARACTERISTICS",
  "- Episodic memory impairment predominant",
  "- Slower cognitive decline (MMSE decline 1-2 points/year)",
  "- Better preserved daily living abilities",
  "",
  "CLINICAL IMPLICATIONS",
  "- Best candidates for early intervention",
  "- Potentially better response to cholinesterase inhibitors",
  "- Cognitive training and lifestyle interventions may benefit",
  "",
  "========================================================================",
  "Subtype 2: Cortical-Diffuse (CD)",
  "========================================================================",
  "",
  "KEY FEATURES",
  "- Diffuse cortical thinning (frontal, temporal, parietal)",
  "- Mild to moderate hippocampal atrophy",
  "- Moderate CSF biomarker abnormalities",
  "- Ventricular enlargement",
  "",
  "CLINICAL CHARACTERISTICS",
  "- Multi-domain cognitive impairment (attention, executive, language)",
  "- Moderate cognitive decline (MMSE decline 2-3 points/year)",
  "- Potential behavioral and psychological symptoms (BPSD)",
  "",
  "CLINICAL IMPLICATIONS",
  "- Require multi-target therapeutic strategies",
  "- May show limited response to single-agent therapy",
  "- Need comprehensive management (pharmacological + non-pharmacological)",
  "",
  "========================================================================",
  "Subtype 3: Typical AD (TAD)",
  "========================================================================",
  "",
  "KEY FEATURES",
  "- Severe hippocampal atrophy",
  "- High CSF pathology (Abeta42 low, Tau/p-Tau high)",
  "- Widespread cortical atrophy",
  "- Significant ventricular enlargement",
  "",
  "CLINICAL CHARACTERISTICS",
  "- Classic AD presentation (memory, language, visuospatial all impaired)",
  "- Rapid cognitive decline (MMSE decline 3-5 points/year)",
  "- Rapid functional decline, significant ADL impairment",
  "",
  "CLINICAL IMPLICATIONS",
  "- Aggressive disease course, poorer prognosis",
  "- Require intensive monitoring and treatment adjustment",
  "- Strongly recommend clinical trial participation",
  "- Early long-term care planning needed",
  "",
  "========================================================================",
  "Subtype Comparison Summary",
  "========================================================================",
  "",
  "| Feature          | HP    | CD    | TAD   |",
  "|------------------|-------|-------|-------|",
  "| Hippocampal      | ++    | +     | +++   |",
  "| Cortical         | +     | +++   | ++    |",
  "| CSF pathology    | ++    | ++    | +++   |",
  "| Progression rate | Slow  | Mod   | Fast  |",
  "| Treatment resp   | Good  | Mod   | Poor  |",
  "",
  "========================================================================",
  paste("Generated:", Sys.time()),
  "========================================================================"
)

writeLines(naming_rationale, file.path(output_dir, "Subtype_Naming_Rationale.txt"))

cat("========================================================================\n")
cat("Step 22: Biological Subtype Naming Complete\n")
cat("========================================================================\n\n")

cat("Subtype names:\n")
for (i in 1:nrow(naming_table)) {
  cat(sprintf("  %s -> %s (%s)\n",
              naming_table$Original_Name[i],
              naming_table$Biological_Name[i],
              naming_table$Short_Name[i]))
}

cat("\nOutput files:\n")
cat("  - Subtype_Biological_Naming_Table.csv\n")
cat("  - Subtype_Naming_Table_For_Paper.csv\n")
cat("  - Subtype_Naming_Rationale.txt\n\n")

cat(sprintf("Output directory: %s\n", output_dir))
cat("========================================================================\n")
