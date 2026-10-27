fn main() {
    println!("cargo:rerun-if-changed=assets/rust-agent.ico");
    println!("cargo:rerun-if-env-changed=RC_PATH");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let icon = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
            .join("assets/rust-agent.ico");
        winresource::WindowsResource::new()
            .set_icon(icon.to_str().expect("Icon path must be valid UTF-8"))
            .compile()
            .expect("Failed to compile Windows icon resources; install the Windows SDK");
    }
}
