# In R, set AD_MULTIMODAL_PYTHON to the Python interpreter containing requirements.txt.
# source("/absolute/path/to/AD_Multimodal_Study/run_all.R", encoding="UTF-8")
run_study <- function() {
  frames <- sys.frames()
  files <- lapply(frames,function(x)x$ofile)
  files <- Filter(function(x)!is.null(x)&&length(x)==1L,files)
  if (!length(files)) stop("Run this file using source() with its absolute path")
  root <- dirname(normalizePath(files[[length(files)]],winslash="/",mustWork=TRUE))
  python <- Sys.getenv("AD_MULTIMODAL_PYTHON",unset="")
  if (!nzchar(python)) python <- Sys.which("python")
  if (!nzchar(python)) stop("Set AD_MULTIMODAL_PYTHON to the full Python executable path")
  rscript <- file.path(R.home("bin"),if(.Platform$OS.type=="windows")"Rscript.exe" else "Rscript")
  status <- system2(python,shQuote(c(file.path(root,"analysis_pipeline/run_analysis_pipeline.py"),
    "--config",file.path(root,"analysis_config.json"),"--rscript",rscript)))
  if (status!=0L) stop("Study pipeline stopped. Inspect the named stage log in results/logs")
}
run_study()
