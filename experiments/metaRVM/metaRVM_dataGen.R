# install.packages("devtools", repos = "https://cloud.r-project.org", dependencies = TRUE)
# devtools::install_github("NSF-RESUME/MetaRVM")
library(MetaRVM)

# Accept seed value as command line argument or set default
args <- commandArgs(trailingOnly = TRUE)
user_seed <- if(length(args) > 0) as.numeric(args[1]) else 42
set.seed(user_seed)

# every hour for 0.8 M point

N_pop <- 1 # number of population
N <- 10000 # population size
I0 <- 10 # number of initial infections at the start of the simulation
V0 <- 1000 # number of vaccinated people 
R0 <- 0 # number of people recovered from prior infection
S0 <- N - I0 - V0 - R0

m <- matrix(1, ncol = 1, nrow = 1) # mixing matrix
vac_mat <- matrix(V0, ncol = 1)

run_sim <- function(ts, tv, dv, de, dp, da, ds, dh, dr, ve, nsteps = 100){
  out <- meta_sim(N_pop = N_pop,
                  S0 = S0,
                  I0 = I0,
                  P0 = N,
                  V0 = V0,
                  R0 = R0,
                  m_weekday_day =  m,
                  m_weekend_day = m,
                  m_weekday_night = m,
                  m_weekend_night = m,
                  delta_t = 0.5,
                  tvac = 0,
                  vac_mat = vac_mat,
                  ts = ts,
                  tv = tv,
                  dv = dv,
                  de = de,
                  pea = 0.5,
                  dp = dp,
                  da = da,
                  ds = ds,
                  psr = 0.5,
                  dh = dh,
                  phr = 0.8,
                  dr = dr,
                  ve = ve,
                  nsteps = 100,
                  is.stoch = 0,
                  seed = NULL)
  
  out_H <- get_disease_state(data.frame(out), disease_states = c("n_IsympH"))
  return(sum(out_H$value))  # Normalize by total population
}

# Generate 1M samples (5 hours on 40 cores, 1 hours on 192 cores)
n_samples <- 1000000

# Load parallel processing package
library(parallel)

# Set fixed number of threads (e.g., 4 threads)
n_cores <- 192  # Change this number to your desired number of threads

# Create cluster
cl <- makeCluster(n_cores)

# Calculate how many samples per core (moved this up before clusterExport)
samples_per_core <- ceiling(n_samples / n_cores)

# Export necessary functions and variables to cluster
clusterExport(cl, c("meta_sim", "get_disease_state", "run_sim", 
                    "N_pop", "S0", "I0", "N", "V0", "R0", "m", "vac_mat",
                    "samples_per_core", "user_seed"))  # Added samples_per_core and user_seed here

# Load required packages on each cluster
clusterEvalQ(cl, {
    library(MetaRVM)
})

# Parallel processing function
generate_samples <- function(n) {
    set.seed(user_seed * 192 + n)  # Different seed for each core, but based on user_seed
    local_results <- matrix(nrow = samples_per_core, ncol = 11)
    
    for(i in 1:samples_per_core) {
        # Random parameter sampling
        ts <- runif(1, 0.1, 0.9)
        tv <- runif(1, 0.1, 0.9)
        dv <- runif(1, 30, 90)
        de <- runif(1, 1, 5)
        dp <- runif(1, 1, 3)
        da <- runif(1, 1, 9)
        ds <- runif(1, 1, 9)
        dh <- runif(1, 1, 5)
        dr <- runif(1, 30, 90)
        ve <- runif(1, 0.3, 0.8)
        
        # Store inputs
        local_results[i, 1:10] <- c(ts, tv, dv, de, dp, da, ds, dh, dr, ve)
        
        # Run simulation and store output
        local_results[i, 11] <- run_sim(ts = ts, tv = tv, dv = dv, de = de, 
                                      dp = dp, da = da, ds = ds, dh = dh, 
                                      dr = dr, ve = ve)
    }
    return(local_results)
}

# Start total timer
total_start_time <- Sys.time()

# Start parallel processing timer
parallel_start_time <- Sys.time()

# Run parallel processing
results_list <- parLapply(cl, 1:n_cores, generate_samples)

# Stop cluster
stopCluster(cl)

# Calculate parallel processing time
parallel_end_time <- Sys.time()
parallel_duration <- difftime(parallel_end_time, parallel_start_time, units = "mins")

# Start post-processing timer
post_start_time <- Sys.time()

# Combine results from all cores
results <- do.call(rbind, results_list)
# Trim excess rows if any
results <- results[1:n_samples,]

# Calculate standard deviation of the output (column 11)
output_sd <- sd(results[,11])
# Divide output by its standard deviation and add 1
results[,11] <- (results[,11] / output_sd) + 1

# After generating results matrix, normalize inputs (first 10 columns)
# Create normalization function to scale to [0,1]
normalize <- function(x, min_val, max_val) {
  return((x - min_val) / (max_val - min_val))
}

# Define min and max values for each parameter
param_bounds <- matrix(c(
  0.1, 0.9,  # ts
  0.1, 0.9,  # tv
  30, 90,    # dv
  1, 5,      # de
  1, 3,      # dp
  1, 9,      # da
  1, 9,      # ds
  1, 5,      # dh
  30, 90,    # dr
  0.3, 0.8   # ve
), ncol = 2, byrow = TRUE)

# Normalize each input column
for(i in 1:10) {
  results[,i] <- normalize(results[,i], param_bounds[i,1], param_bounds[i,2])
}

# Randomly shuffle the data
set.seed(user_seed)  # for reproducibility
shuffle_idx <- sample(nrow(results))
results <- results[shuffle_idx,]

# Split into training (90%) and testing (10%)
n_train <- floor(0.9 * nrow(results))
train_data <- results[1:n_train,]
test_data <- results[(n_train+1):nrow(results),]

# Calculate post-processing time
post_end_time <- Sys.time()
post_duration <- difftime(post_end_time, post_start_time, units = "mins")

# Calculate total time
total_end_time <- Sys.time()
total_duration <- difftime(total_end_time, total_start_time, units = "mins")

# Print timing information
cat(sprintf("\nTiming Summary:\n"))
cat(sprintf("Parallel Processing Time: %.2f minutes\n", parallel_duration))
cat(sprintf("Post-processing Time: %.2f minutes\n", post_duration))
cat(sprintf("Total Runtime: %.2f minutes\n", total_duration))

# Create directory with seed in name
dir_name <- sprintf("./metaRVMdata_seed")
dir.create(dir_name)


# Save training and testing data with seed in filename
write.table(train_data, 
            file = file.path(dir_name, sprintf("train_data_seed%d.txt", user_seed)), 
            row.names = FALSE, 
            col.names = FALSE, 
            sep = ",")

write.table(test_data, 
            file = file.path(dir_name, sprintf("test_data_seed%d.txt", user_seed)), 
            row.names = FALSE, 
            col.names = FALSE, 
            sep = ",")