from django.db import models

class SystemLog(models.Model):
    LOG_TYPES = [
        ('USER', 'User Activity'),
        ('AI', 'AI Processing'),
        ('ADMIN', 'Admin Action'),
        ('SYSTEM', 'System Alert'),
    ]
    log_type = models.CharField(max_length=20, choices=LOG_TYPES)
    message = models.TextField()
    user_email = models.EmailField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'system_logs'
        ordering = ['-created_at']
