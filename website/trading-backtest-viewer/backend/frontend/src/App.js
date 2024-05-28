import React from 'react';
import { Container, AppBar, Typography, Toolbar, Button, TextField, Grid } from '@mui/material';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import UploadForm from './components/UploadForm';
import Results from './components/Results';

function App() {
    return (
        <Router>
            <Container maxWidth="lg">
                <AppBar position="static">
                    <Toolbar>
                        <Typography variant="h6" style={{ flexGrow: 1 }}>
                            Trading Backtest Viewer
                        </Typography>
                        <Button color="inherit" component={Link} to="/">Upload</Button>
                        <Button color="inherit" component={Link} to="/results">Results</Button>
                    </Toolbar>
                </AppBar>
                <Routes>
                    <Route exact path="/" component={UploadForm} />
                    <Route path="/results" component={Results} />
                </Routes>
            </Container>
        </Router>
    );
}

export default App;
