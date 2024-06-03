import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
import { Button, Box, Typography, TextField, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Collapse } from '@mui/material';
import API from '../api';
import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc';
import timezone from 'dayjs/plugin/timezone';

dayjs.extend(utc);
dayjs.extend(timezone);

const BATCH_SIZE = 1000;  // Adjust this value based on your requirements and server capability

function UploadForm() {
  const [username, setUsername] = useState('');
  const [message, setMessage] = useState('');
  const [results, setResults] = useState([]);
  const [openRow, setOpenRow] = useState(null);

  const handleUsernameChange = (e) => setUsername(e.target.value);

  const calculateMarketMinutes = (start, minutesTaken) => {
    let totalMinutes = 0;
    let currentTime = dayjs(start).tz('America/New_York');
    
    while (totalMinutes < minutesTaken) {
      if (currentTime.hour() >= 9 && currentTime.hour() < 16 && currentTime.day() >= 1 && currentTime.day() <= 5) {
        totalMinutes++;
      }
      currentTime = currentTime.add(1, 'minute');
    }
    
    return currentTime;
  };
  
  const findSaleIndex = (dates, createdAt, minutesTaken) => {
    const startDateTime = dayjs(createdAt).tz('America/New_York');
    const endDateTime = calculateMarketMinutes(startDateTime, minutesTaken);
    return findNearestIndex(dates, endDateTime.format('YYYY-MM-DD HH:mm:ss'));
  };

  const handleUpload = () => {
    if (!username) {
      setMessage('Please provide a username.');
      return;
    }
    const data = { username };
    API.post('upload/', data)
      .then(response => {
        setMessage('Processing successful');
        const initialResults = response.data.data || [];
        initialResults.forEach(result => {
          API.get(`/api/stockdata/${result.ticker}/`)
            .then(chartResponse => {
              result.chartData = chartResponse.data.chartData;
              result.createdAtIndex = findNearestIndex(chartResponse.data.chartData.dates, result.created_at);
              result.saleIndex = findSaleIndex(chartResponse.data.chartData.dates, result.created_at, result.minutes_taken);
              setResults(prevResults => [...prevResults, result]);
              if (chartResponse.data.chartData) {
                handleBatchUpload(result.ticker, chartResponse.data.chartData);
              }
            })
            .catch(error => {
              console.error('Error fetching chart data:', error);
            });
        });
      })
      .catch(error => {
        setMessage('Error processing username');
        console.error('Upload error:', error);
      });
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

  const handleBatchUpload = (ticker, chartData) => {
    const stockData = chartData.dates.map((date, index) => ({
      date: date,
      close: chartData.prices[index],
    }));
  
    const batches = [];
    for (let i = 0; i < stockData.length; i += BATCH_SIZE) {
      const batch = stockData.slice(i, i + BATCH_SIZE);
      batches.push(batch);
    }
  
    Promise.all(batches.map(batch => 
      fetch('/api/batch-upload/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': window.csrfToken // Include the CSRF token in the headers
        },
        body: JSON.stringify({ ticker, stock_data: batch })
      })
    ))
    .then(() => {
      console.log('All batches uploaded successfully');
    })
    .catch(error => {
      console.error('Error uploading batches:', error);
    });
  };

  const toggleRow = (index) => {
    setOpenRow(openRow === index ? null : index);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" component="h2" style={{ paddingBottom: '16px' }}>
        Process Heisenberg_100k
      </Typography>
      <TextField
        label="Twitter Username"
        value={username}
        onChange={handleUsernameChange}
        fullWidth
        style={{ marginBottom: '16px' }}
      />
      <Button variant="contained" color="primary" onClick={handleUpload} style={{ marginTop: '16px' }}>
        Process
      </Button>
      {message && (
        <Typography variant="body2" color="error" style={{ marginTop: '16px' }}>
          {message}
        </Typography>
      )}
      {results.length > 0 && (
        <TableContainer component={Paper} style={{ marginTop: '16px' }}>
          <Table>
            <TableHead>
              <TableRow>
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
                  <TableRow onClick={() => toggleRow(index)} style={{ cursor: 'pointer' }}>
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
                  <TableRow>
                    <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={5}>
                      <Collapse in={openRow === index} timeout="auto" unmountOnExit>
                        <Box sx={{ margin: 1 }}>
                          <Typography variant="h6" gutterBottom component="div">
                            Detailed Chart
                          </Typography>
                          {result.chartData && (
                            <>
                              <Line data={{
                                labels: result.chartData.dates.map(date => dayjs(date).tz('America/New_York').format('MM-DD hh:mm A')) || [],
                                datasets: [
                                  {
                                    label: 'Price',
                                    data: result.chartData.prices || [],
                                    borderColor: 'rgb(75, 192, 192)',
                                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                    order: 2,  // Ensure this dataset is drawn below
                                  },
                                  {
                                    label: 'Created At',
                                    data: result.chartData.dates.map((date, i) =>
                                      i === result.createdAtIndex ? result.chartData.prices[i] : null
                                    ),
                                    borderColor: 'rgb(255, 99, 132)',
                                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                    pointRadius: result.chartData.dates.map((date, i) =>
                                      i === result.createdAtIndex ? 7 : 0
                                    ),
                                    showLine: false,
                                    order: 1,  // Ensure this dataset is drawn above
                                  },
                                  {
                                    label: 'Sale Point',
                                    data: result.chartData.dates.map((date, i) =>
                                      i === result.saleIndex ? result.chartData.prices[i] : null
                                    ),
                                    borderColor: 'rgb(0, 255, 0)',
                                    backgroundColor: 'rgba(0, 255, 0, 0.5)',
                                    pointRadius: result.chartData.dates.map((date, i) =>
                                      i === result.saleIndex ? 7 : 0
                                    ),
                                    showLine: false,
                                    order: 1,  // Ensure this dataset is drawn above
                                  }
                                ]
                              }} />
                            </>
                          )}
                        </Box>
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
}

export default UploadForm;
