# Generated by Django 2.1 on 2019-07-04 13:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0004_auto_20190701_1731'),
    ]

    operations = [
        migrations.CreateModel(
            name='Viewer',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('car_number', models.CharField(default=0, max_length=10)),
                ('in_time', models.TimeField()),
                ('out_time', models.TimeField()),
            ],
        ),
        migrations.AlterField(
            model_name='profile',
            name='lat',
            field=models.FloatField(default=10.9925),
        ),
        migrations.AlterField(
            model_name='profile',
            name='lng',
            field=models.FloatField(default=76.9614),
        ),
    ]
