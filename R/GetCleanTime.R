
get_clean_time <- function(df, timeframe){
  
  data <- df %>%
    mutate(
      Glpostdt = lubridate::floor_date(Glpostdt, paste0(timeframe)),
      ID = paste0(Custname,Itemnmbr)) %>% 
    filter(ID %in% last_trans_date_clean$ID)
  
}