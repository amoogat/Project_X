import React, { useState, useEffect } from 'react';
import API from '../api';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography } from '@mui/material';

function Results() {
    const [results, setResults] = useState([]);

    useEffect(() => {
        API.get('results/')
            .then(response => {
                setResults(response.data);
            })
            .catch(error => {
                console.error('Error fetching results:', error);
            });
    }, []);

    return (
        <TableContainer component={Paper}>
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
    );
}

export default Results;
