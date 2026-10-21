# Restricted cubic spline logistic models in R (rms), the same models as analysis/splines.py.
# Usage, from the project folder:  Rscript analysis/splines.R

library(rms)

df <- read.csv("data/analysis_dataset.csv")
dd <- datadist(df); options(datadist = "dd")

covariates <- c("cpss_helplessness", "phq8_total", "gad10_total", "screen_time",
                "trauma_count", "pa_level", "gender_female", "education_years")

fit_spline <- function(target) {
    adjust <- setdiff(covariates, target)
    f <- as.formula(paste0("outcome_si_binary ~ rcs(", target, ", 5) + ", paste(adjust, collapse = " + ")))
    fit <- lrm(f, data = df, x = TRUE, y = TRUE)
    a <- anova(fit)
    i <- which(rownames(a) == target)
    j <- which(trimws(rownames(a)) == "Nonlinear")
    j <- j[j > i][1]
    data.frame(variable = target, overall_p = a[i, "P"], nonlinearity_p = a[j, "P"], C = unname(fit$stats["C"]))
}

targets <- c("phq8_total", "cpss_helplessness", "gad10_total", "cpss_self_efficacy")
results <- do.call(rbind, lapply(targets, fit_spline))
print(results, digits = 4)
dir.create("results/splines", showWarnings = FALSE, recursive = TRUE)
write.csv(results, "results/splines/spline_tests_R.csv", row.names = FALSE)
