library(tidyverse)
library(readxl)

Path <- "Data/Demand planner 092820.xlsx"

Sheets <- readxl::excel_sheets(Path)
# cut out first sheet
Sheets <- Sheets[2:length(Sheets)]


Data <- tibble()

for (sheet in Sheets) {

  # appends each individual sheet to a tibble

  Item <- readxl::read_xlsx(Path, sheet, range = cell_cols("A:F"), col_types = c("text")) %>%
    mutate(ItemName = sheet)


  Data <- Data %>%
    bind_rows(., Item)
}


# cleaning NAs
CleanData <- Data %>%
  mutate(
    Date = as.numeric(Date),
    Date = janitor::excel_numeric_to_date(Date)
  ) %>%
  select(ItemName, Date, Sales = `Actual Sales`) %>%
  filter(!is.na(Date) & !is.na(Sales)) %>%
  mutate(Sales = as.numeric(Sales))


GetMaxDate <- function(df) {
  # see which are discontinued
  max(df$Date)
}



# processing
ProcessedData <- CleanData %>%
  tsibble::tsibble(key = "ItemName", index = "Date") %>%
  tsibble::fill_gaps(Sales = 0L) %>%
  tibble() %>%
  nest(TimeData = c("Date", "Sales")) %>%
  mutate(MaxDate = map(.x = TimeData, .f = GetMaxDate)) %>%
  unnest(MaxDate) %>%
  filter(MaxDate >= "2020-01-01") %>%
  unnest(TimeData)

ProcessedData %>%
  mutate(Month = lubridate::floor_date(Date, "month")) %>%
  group_by(Month, ItemName) %>%
  summarise(Sales = sum(Sales, na.rm = TRUE)) %>%
  ggplot(aes(Month, Sales, group = ItemName)) +
  geom_line(size = 1) +
  facet_wrap(~ItemName, scales = "free")

plotly::ggplotly()
