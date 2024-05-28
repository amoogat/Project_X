import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import UploadForm from './components/UploadForm';
import Results from './components/Results';
import { AppBar, Toolbar, Typography, Button, Container } from '@mui/material';

function App() {
    return (
        <Router>
            <div>
                <AppBar position="static">
                    <Toolbar>
                        <Typography variant="h6" style={{ flexGrow: 1 }}>
                            Trading Backtest Viewer
                        </Typography>
                        <Button color="inherit" component={Link} to="/upload">Upload</Button>
                        <Button color="inherit" component={Link} to="/results">Results</Button>
                    </Toolbar>
                </AppBar>
                <Container>
                    <Routes>
                        <Route path="/upload" element={<UploadForm />} />
                        <Route path="/results" element={<Results />} />
                        <Route path="/" element={<UploadForm />} />
                    </Routes>
                </Container>
            </div>
        </Router>
    );
}

export default App;
