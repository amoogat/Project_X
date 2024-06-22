import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
import {
  Button, Box, Typography, TextField, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Paper, Collapse, CircularProgress, Card, CardContent
} from '@mui/material';
import API from '../api';
import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc';
import timezone from 'dayjs/plugin/timezone';
import theme from './../theme';


dayjs.extend(utc);
dayjs.extend(timezone);

const UploadForm = () => {
  const [username, setUsername] = useState('');
  const [message, setMessage] = useState('');
  const [results, setResults] = useState([]);
  const [openRow, setOpenRow] = useState(null);
  const [portfolioChartData, setPortfolioChartData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUsernameChange = (e) => setUsername(e.target.value);

  const handleUpload = async () => {
    if (!username) {
      setMessage('Please provide a username.');
      return;
    }
    setLoading(true);
    setMessage('');
    setResults([]);
    setPortfolioChartData(null);
    const data = { username };

    try {
      const response = await API.post('upload/', data);
      setMessage('Processing successful');
      const initialResults = response.data.data || [];
      setPortfolioChartData(response.data.portfolio_chart_data);

      const updatedResults = await Promise.all(initialResults.map(async result => {
        try {
          const chartResponse = await API.get(`/api/stockdata/${result.ticker}/`);
          result.chartData = chartResponse.data.chartData;
          result.createdAtIndex = findNearestIndex(result.chartData.dates, result.created_at);
          result.saleIndex = findNearestIndex(result.chartData.dates, result.sold_at_date);
          result.tweet_text = cleanString(chartResponse.data.tweet_text);
          return result;
        } catch (error) {
          console.error('Error fetching chart data:', error);
          return null; 
        }
      }));

      // Update results state once after processing all initial results
      setResults(updatedResults.filter(result => result !== null));

    } catch (error) {
      setMessage('Error processing username');
      console.error('Upload error:', error);
    } finally {
      setLoading(false);
    }
  };

  const findNearestIndex = (dates, targetDate) => {
    const targetTime = dayjs(targetDate).tz('America/New_York').valueOf();
    let nearestIndex = 0;
    let minDiff = Infinity;

    dates.forEach((date, index) => {
      const dateTime = dayjs(date).tz('America/New_York').valueOf();
      const diff = Math.abs(dateTime - targetTime);
      if (diff < minDiff) {
        minDiff = diff;
        nearestIndex = index;
      }
    });

    return nearestIndex;
  };

  const cleanString = (input) => {
    if (typeof input === 'string' && input.startsWith("b'")) {
      return input.slice(2, -1).replace(/\\'/g, "'");
    }
    return input;
  };

  const toggleRow = (index) => {
    setOpenRow(openRow === index ? null : index);
  };

  return (
    <Box sx={{ p: 2, bgcolor: 'background.default', color: 'text.primary' }}>
      <Typography variant="h5" component="h2" style={{ paddingBottom: '16px' }}>
        Process Twitter Username
      </Typography>
      <TextField
        label="Twitter Username"
        value={username}
        onChange={handleUsernameChange}
        fullWidth
        sx={{ marginBottom: '16px', backgroundColor: 'background.paper' }}
        InputLabelProps={{
          style: { color: theme.palette.secondary.main }
        }}
      />
      <Button variant="contained" color="primary" onClick={handleUpload} style={{ marginTop: '16px' }} disabled={loading}>
        Process
      </Button>
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
          <CircularProgress />
        </Box>
      )}
      {message && (
        <Typography variant="body2" color={message.includes('error') ? 'error' : 'primary'} style={{ marginTop: '16px' }}>
          {message}
        </Typography>
      )}
      {portfolioChartData && (
        <Box sx={{ marginTop: '16px' }}>
          <Typography variant="h6" gutterBottom component="div">
            Portfolio Performance
          </Typography>
          <Card>
            <CardContent style={{ height: '500px' }}>
              <Line
                data={{
                  labels: portfolioChartData.dates.map(date => dayjs(date).tz('America/New_York').format('MM-DD hh:mm A')),
                  datasets: [
                    {
                      label: 'Portfolio Value',
                      data: portfolioChartData.values,
                      borderColor: '#4A90E2',
                      backgroundColor: 'rgba(74, 144, 226, 0.5)',
                      borderWidth: 2,
                      pointBackgroundColor: '#FFFFFF',
                      pointBorderColor: '#4A90E2',
                      pointHoverBackgroundColor: '#4A90E2',
                      pointHoverBorderColor: '#FFFFFF',
                      tension: 0.4
                    }
                  ]
                }}
                options={{
                  maintainAspectRatio: true,
                  responsive: true,
                  plugins: {
                    tooltip: {
                      enabled: true,
                      mode: 'index',
                      intersect: false,
                      callbacks: {
                        label: function(tooltipItems) {
                          return `Value: ${tooltipItems.raw.toFixed(2)}`;
                        }
                      }
                    }
                  },
                  hover: {
                    mode: 'nearest',
                    intersect: true
                  },
                  elements: {
                    line: {
                      tension: 0.4, 
                      borderWidth: 2,
                      borderColor: '#4A90E2',
                      backgroundColor: 'rgba(74, 144, 226, 0.2)',
                    },
                    point: {
                      radius: 0,
                      hoverRadius: 5,
                    }
                  },
                  scales: {
                    x: {
                      grid: {
                        color: 'rgba(255,255,255,0.1)' 
                      },
                      ticks: {
                        autoSkip: true,
                        maxTicksLimit: 15
                      }
                    },
                    y: {
                      grid: {
                        color: 'rgba(255,255,255,0.1)'
                      },
                      beginAtZero: false
                    }
                  }
                }}
              />
            </CardContent>
          </Card>
        </Box>
      )}
      {results.length > 0 && (
        <TableContainer component={Paper} style={{ marginTop: '16px' }}>
          <Table>
            <TableHead>
              <TableRow style={{ backgroundColor: '#282c34', borderBottom: '1px solid #444' }}>
                <TableCell>Ticker</TableCell>
                <TableCell>Created At</TableCell>
                <TableCell>Minutes Taken</TableCell>
                <TableCell>Total Return</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {results.map((result, index) => (
                <React.Fragment key={index}>
                  <TableRow onClick={() => toggleRow(index)} style={{ cursor: 'pointer', backgroundColor: '#333', borderBottom: '1px solid #444' }}>
                    <TableCell component="th" scope="row">{result.ticker || 'N/A'}</TableCell>
                    <TableCell>{dayjs(result.created_at).tz('America/New_York').format('MM-DD hh:mm A') || 'Unknown Date'}</TableCell>
                    <TableCell>{result.minutes_taken || 'N/A'}</TableCell>
                    <TableCell>{result.total_return ? `${result.total_return.toFixed(2)}%` : 'N/A'}</TableCell>
                    <TableCell>
                      <Button>
                        {openRow === index ? 'Hide Details' : 'Show Details'}
                      </Button>
                    </TableCell>
                  </TableRow>
                  <TableRow style={{ backgroundColor: '#282c34' }}>                    
                    <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={5}>
                      <Collapse in={openRow === index} timeout="auto" unmountOnExit>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h6" gutterBottom component="div">
                              Detailed Chart
                            </Typography>
                            {result.chartData && (
                              <>
                                <Line data={{
                                  labels: result.chartData.dates.map(date => dayjs(date).tz('America/New_York').format('MM-DD hh:mm A')),
                                  datasets: [
                                    {
                                      label: 'Price',
                                      data: result.chartData.prices,
                                      borderColor: 'rgb(75, 192, 192)',
                                      backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                      order: 2,
                                    },
                                    {
                                      label: 'Created At',
                                      data: result.chartData.dates.map((date, i) =>
                                        i === result.createdAtIndex ? result.chartData.prices[i] : null
                                      ),
                                      borderColor: 'rgb(255, 99, 132)',
                                      backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                      pointRadius: result.chartData.dates.map((date, i) =>
                                        i === result.createdAtIndex ? 17 : 0
                                      ),
                                      showLine: false,
                                      order: 1,
                                    },
                                    { 
                                      label: 'Sale Point',
                                      data: result.chartData.dates.map((date, i) =>
                                        i === result.saleIndex ? result.chartData.prices[i] : null
                                      ),
                                      borderColor: 'rgb(0, 255, 0)',
                                      backgroundColor: 'rgba(0, 255, 0, 0.5)',
                                      pointRadius: result.chartData.dates.map((date, i) =>
                                        i === result.saleIndex ? 17 : 0
                                      ),
                                      showLine: false,
                                      order: 1,
                                    }
                                  ]
                                }} />
                                <Typography
                                  variant="body2"
                                  style={{
                                    marginTop: '16px',
                                    whiteSpace: 'pre-line',
                                    color: '#424242', // dark grey color
                                    fontWeight: 'normal', // normal weight
                                    backgroundColor: '#f5f5f5', // light grey background
                                    padding: '10px', // adds padding around the text
                                    borderRadius: '4px', // slightly rounded corners for the background
                                    fontFamily: 'Arial, sans-serif' // simpler font style
                                  }}
                                >
                                  {result.tweet_text}
                                </Typography>
                              </>
                            )}
                          </CardContent>
                        </Card>
                      </Collapse>
                    </TableCell>
                  </TableRow>
                </React.Fragment>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default UploadForm;
