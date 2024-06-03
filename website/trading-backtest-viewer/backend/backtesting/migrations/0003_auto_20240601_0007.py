# Generated by Django 3.2.25 on 2024-06-01 04:07

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backtesting', '0002_auto_20240531_2353'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stockdata',
            name='close',
            field=models.DecimalField(decimal_places=2, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='date',
            field=models.DateTimeField(blank=True, default=datetime.datetime.now),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='high',
            field=models.DecimalField(decimal_places=2, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='low',
            field=models.DecimalField(decimal_places=2, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='open',
            field=models.DecimalField(decimal_places=2, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='ticker',
            field=models.CharField(default=' ', max_length=10),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='volume',
            field=models.BigIntegerField(default=0),
        ),
    ]
