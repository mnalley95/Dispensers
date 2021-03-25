###Dispenser Forecasting
source("Dispensers/R/DispenserProcessing.R")
library(ForecastHelpers)
library(tidyverse)
library(tsibble)
library(fable)
library(fabletools)
library(lubridate)

dispensers <- data_month %>% 
  group_by(Glpostdt, Itemnmbr, Custname) %>% 
  summarise(QtyMonth = sum(Quantity, na.rm = TRUE)) %>% 
  rename(CustomerBucket = Custname, ItemNumber = Itemnmbr, ReqMonth = Glpostdt) %>% 
  mutate(ReqMonth = as.Date(ReqMonth)) %>% 
  ungroup() %>% 
  select(CustomerBucket, ItemNumber, ReqMonth, QtyMonth)

dispensers_train <- dispensers %>% 
  nest(cols = c(ReqMonth, QtyMonth)) %>% 
  mutate(rows = map(.x=cols, .f=NROW)) %>% 
  unnest(rows) %>% 
  filter(rows > 6) %>%
  mutate(splits = map(cols, rsample::initial_time_split, 0.9), 
         training = map(splits, rsample::training)
  ) %>% 
  unnest(training) %>% 
  select(CustomerBucket, ItemNumber, ReqMonth, QtyMonth)

dispensers_ts <- ts_format(dispensers, 'Retail', current_month)

dispensers_train <- ts_format(dispensers_train, 'Retail', current_month)

start <- Sys.time()

fit_dispensers_train <- forecast_function(dispensers_train)

fc_dispensers_train <- forecast(fit_dispensers_train, h = 10)


acc_dispensers <- accuracy(
  fc_dispensers_train,
  dispensers_ts,
  measures = list(
    point_accuracy_measures
  )
)

end <- Sys.time()


total_fcst <-
  model_choice_revamp(acc_dispensers,
                      fit_dispensers_train,
                      dispensers_ts,
                      training_month)



ts_grapher <- function(Loc, ItemNumber = NA, dispenser_ts = dispensers_ts){
  
  if(!is.na(ItemNumber)){
  
    dispenser_ts <- dispenser_ts %>% 
      filter(Loc == paste0(Loc) & ItemNumber==paste0(ItemNumber)) 
    
    total_fcst %>% 
    filter(Loc == paste0(Loc) & ItemNumber==paste0(ItemNumber)) %>% 
    autoplot(dispensers_ts, level = NULL) +
    scale_y_continuous(labels = scales::comma)
  } else if(is.na(ItemNumber)) {
    
    dispenser_ts <- dispenser_ts %>% 
      filter(Loc == paste0(Loc)) 
    
    total_fcst %>%
      filter(Loc == paste0(Loc)) %>%
      autoplot(dispensers_ts, level = NULL) +
      scale_y_continuous(labels = scales::comma) +
      facet_wrap(~ItemNumber, scales = 'free_y')


  }
}


dispensers_output <- forecast_output(dispensers_ts, total_fcst, current_month)

dispensers_output <- dispensers_output %>% 
  left_join(., dispenser_items, by = c("ItemNumber" = 'ITEMNMBR')) %>% 
  mutate(Channel = case_when(grepl(".com", Loc) ~ "COM",
                             grepl("di", Loc, ignore.case = TRUE) | grepl("Direct Import", Loc, ignore.case = TRUE) ~ 'DI',
         TRUE ~ 'Domestic'))


saveRDS(total_fcst, file = 'Dispensers/fc_dispensers.RDS')
saveRDS(fc_dispensers_train, file = 'Dispensers/fc_dispensers_train.RDS')
# saveRDS(fit_dispensers, file = 'fit_dispensers.RDS')
saveRDS(fit_dispensers_train, file = 'Dispensers/fit_dispensers_train.RDS')
saveRDS(acc_dispensers, file = 'Dispensers/acc_dispensers.RDS')
saveRDS(dispensers_output, file = 'Dispensers/dispensers_output.RDS')

write_csv(dispensers_output, "Dispensers/Modeling/DispensersOutput.csv")
