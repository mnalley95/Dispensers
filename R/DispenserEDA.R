source("Dispensers/R/DispenserProcessing.R")

data_month %>% 
  ggplot(aes(Glpostdt, Quantity, color = Custname)) +
  scale_y_continuous(labels = scales::comma) +
  geom_line()

data_week %>% 
  group_by(Custname, Glpostdt, Itemnmbr) %>%
  summarise(Quantity = sum(Quantity, na.rm = TRUE)) %>% 
  ggplot(aes(Glpostdt, Quantity, group = Itemnmbr)) +
  scale_y_continuous(labels = scales::comma) +
  geom_line() +
  facet_wrap(~Custname,scales = 'free_y') +
  labs(title = 'Top items by customer')

plotly::ggplotly()

data_month %>% 
  group_by(Custname, Glpostdt, Itemnmbr) %>%
  summarise(Quantity = sum(Quantity, na.rm = TRUE)) %>% 
  ggplot(aes(Glpostdt, Quantity, group = Itemnmbr)) +
  scale_y_continuous(labels = scales::comma) +
  geom_line() +
  facet_wrap(~Custname,scales = 'free_y') +
  labs(title = 'Top items by customer')

plotly::ggplotly()
