import React, { useState } from 'react';
import { TextField, Button, Grid, Typography } from '@mui/material';
import axios from 'axios';

function UploadForm() {
    const [twitterUrl, setTwitterUrl] = useState('');

    const handleUpload = async () => {
        try {
            const response = await axios.post('http://localhost:8000/api/strategies/twitter_url/', { twitter_url: twitterUrl });
            alert('Twitter URL processed successfully');
        } catch (error) {
            console.error('Error uploading Twitter URL:', error);
            alert('Failed to process Twitter URL');
        }
    };

    return (
        <Grid container spacing={3} alignItems="center">
            <Grid item xs={12}>
                <Typography variant="h5">Upload Twitter URL</Typography>
            </Grid>
            <Grid item xs={12}>
                <TextField
                    fullWidth
                    label="Twitter URL"
                    value={twitterUrl}
                    onChange={(e) => setTwitterUrl(e.target.value)}
                />
            </Grid>
            <Grid item xs={12}>
                <Button variant="contained" color="primary" onClick={handleUpload}>
                    Upload
                </Button>
            </Grid>
        </Grid>
    );
}

export default UploadForm;
