import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
import {
  Button, Box, Typography, TextField, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Paper, Collapse, CircularProgress
} from '@mui/material';
import API from '../api';
import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc';
import timezone from 'dayjs/plugin/timezone';

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
          result.tweet_text = chartResponse.data.tweet_text;
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

  const toggleRow = (index) => {
    setOpenRow(openRow === index ? null : index);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" component="h2" style={{ paddingBottom: '16px' }}>
        Process Twitter Username
      </Typography>
      <TextField
        label="Twitter Username"
        value={username}
        onChange={handleUsernameChange}
        fullWidth
        style={{ marginBottom: '16px' }}
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
          <Line data={{
            labels: portfolioChartData.dates.map(date => dayjs(date).tz('America/New_York').format('MM-DD hh:mm A')),
            datasets: [
              {
                label: 'Portfolio Value',
                data: portfolioChartData.values,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
              }
            ]
          }} />
        </Box>
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
                              <Typography variant="body2" style={{ marginTop: '16px', whiteSpace: 'pre-line' }}>
                                {result.tweet_text}
                              </Typography>
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