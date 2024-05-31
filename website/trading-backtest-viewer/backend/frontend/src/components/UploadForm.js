import React, { useState } from 'react';
import API from '../api';
import { Button, Box, Typography, TextField, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';

function UploadForm() {
    const [username, setUsername] = useState('');
    const [message, setMessage] = useState('');
    const [results, setResults] = useState([]);

    const handleUsernameChange = (e) => {
        setUsername(e.target.value);
    };

    const handleUpload = () => {
        if (!username) {
            setMessage('Please provide a username.');
            return;
        }

        const data = { username: username };

        API.post('upload/', data)
            .then(response => {
                setMessage('Processing successful');
                console.log('Response data:', response.data);
                setResults(response.data.data || []);
            })
            .catch(error => {
                setMessage('Error processing username');
                console.error('Error processing username:', error);
                if (error.response) {
                    console.error('Error response data:', error.response.data);
                }
            });
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
                                <TableCell>ATR Multiplier</TableCell>
                                <TableCell>Trailing Stop Multiplier</TableCell>
                                <TableCell>ATR Period</TableCell>
                                <TableCell>Total Return</TableCell>
                                <TableCell>Portfolio Variance</TableCell>
                                <TableCell>Sharpe Ratio</TableCell>
                                <TableCell>Final Equity</TableCell>
                                <TableCell>Maximum Drawdown</TableCell>
                                <TableCell>Successful Trades</TableCell>
                                <TableCell>Minutes Taken</TableCell>
                                <TableCell>Score</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {results.map((result, index) => (
                                <TableRow key={index}>
                                    <TableCell>{result.ticker}</TableCell>
                                    <TableCell>{new Date(result.created_at).toLocaleString()}</TableCell>
                                    <TableCell>{result.atr_multiplier}</TableCell>
                                    <TableCell>{result.trailing_stop_multiplier}</TableCell>
                                    <TableCell>{result.atr_period}</TableCell>
                                    <TableCell>{result.total_return}</TableCell>
                                    <TableCell>{result.portfolio_variance}</TableCell>
                                    <TableCell>{result.sharpe_ratio}</TableCell>
                                    <TableCell>{result.final_equity}</TableCell>
                                    <TableCell>{result.maximum_drawdown}</TableCell>
                                    <TableCell>{result.successful_trades}</TableCell>
                                    <TableCell>{result.minutes_taken}</TableCell>
                                    <TableCell>{result.score}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            )}
        </Box>
    );
}

export default UploadForm;
