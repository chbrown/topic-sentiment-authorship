import('ggplot2')
import('scales')
#options(max.print=1000)

all_tweet_dates = read.table('~/github/tsa/R/TweetTimes.csv', header=TRUE)

dates = strptime(all_tweet_dates$TweetTime, format="%Y-%m-%d")
dates = as.POSIXct(all_tweet_dates$TweetTime, format="%Y-%m-%d")
dates = as.Date(all_tweet_dates$TweetTime, format="%Y-%m-%d")

tail(dates)
qplot(dates, aes(x=dates))

plot(xtabs(~ dates))

hist(dates, breaks=50, density=FALSE)
#, breaks=date_breaks("1 day")

qplot(dates, geom='histogram', binwidth=1) +
  scale_x_date(labels=date_format("%m-%d"), breaks=date_breaks("month"))

# ggplot(as.integer(dates)) +
  # geom_histogram(binwidth=1)

# scale_x_date(labels=date_format("%Y-%m-%d"))
