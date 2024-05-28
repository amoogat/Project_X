import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Grid, Typography, Paper, List, ListItem, ListItemText } from '@mui/material';

function Results() {
    const [results, setResults] = useState([]);

    useEffect(() => {
        const fetchResults = async () => {
            try {
                const response = await axios.get('http://localhost:8000/api/backtestresults/');
                setResults(response.data);
            } catch (error) {
                console.error('Error fetching results:', error);
            }
        };

        fetchResults();
    }, []);

    return (
        <Grid container spacing={3}>
            <Grid item xs={12}>
                <Typography variant="h5">Backtest Results</Typography>
            </Grid>
            <Grid item xs={12}>
                <Paper>
                    <List>
                        {results.map((result, index) => (
                            <ListItem key={index}>
                                <ListItemText
                                    primary={`Ticker: ${result.ticker}, Total Return: ${result.total_return}`}
                                    secondary={`Created At: ${result.created_at}, Sharpe Ratio: ${result.sharpe_ratio}`}
                                />
                            </ListItem>
                        ))}
                    </List>
                </Paper>
            </Grid>
        </Grid>
    );
}

export default Results;
