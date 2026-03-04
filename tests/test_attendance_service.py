from analyzing_faces.services.attendance import AttendanceService


def test_attendance_file_bootstrap(tmp_path):
    file_path = tmp_path / "attendance.csv"
    service = AttendanceService(file_path)
    assert file_path.exists()
    assert service.list_events() == []
