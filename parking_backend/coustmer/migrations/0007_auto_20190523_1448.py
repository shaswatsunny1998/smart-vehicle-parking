# -*- coding: utf-8 -*-
# Generated by Django 1.9 on 2019-05-23 09:18
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('coustmer', '0006_userprofileinfo_totalcarsp'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userprofileinfo',
            name='totalcarsp',
            field=models.BigIntegerField(default='00'),
        ),
    ]