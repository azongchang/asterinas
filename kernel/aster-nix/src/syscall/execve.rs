// SPDX-License-Identifier: MPL-2.0

use aster_frame::{cpu::UserContext, user::UserContextApi};
use aster_rights::WriteOp;

use super::{constants::*, SyscallReturn};
use crate::{
    cpu::LinuxAbi,
    fs::{
        file_table::FileDesc,
        fs_resolver::{FsPath, AT_FDCWD},
        path::Dentry,
        utils::InodeType,
    },
    prelude::*,
    process::{
        check_executable_file, credentials_mut, load_program_to_vm,
        posix_thread::{PosixThreadExt, ThreadName},
        Credentials, MAX_ARGV_NUMBER, MAX_ARG_LEN, MAX_ENVP_NUMBER, MAX_ENV_LEN,
    },
    util::{read_cstring_from_user, read_val_from_user},
};

pub fn sys_execve(
    filename_ptr: Vaddr,
    argv_ptr_ptr: Vaddr,
    envp_ptr_ptr: Vaddr,
    context: &mut UserContext,
) -> Result<SyscallReturn> {
    let elf_file = {
        let executable_path = read_filename(filename_ptr)?;
        lookup_executable_file(AT_FDCWD, executable_path, OpenFlags::empty())?
    };

    do_execve(elf_file, argv_ptr_ptr, envp_ptr_ptr, context)?;
    Ok(SyscallReturn::NoReturn)
}

pub fn sys_execveat(
    dfd: FileDesc,
    filename_ptr: Vaddr,
    argv_ptr_ptr: Vaddr,
    envp_ptr_ptr: Vaddr,
    flags: u32,
    context: &mut UserContext,
) -> Result<SyscallReturn> {
    let elf_file = {
        let flags = OpenFlags::from_bits_truncate(flags);
        let filename = read_filename(filename_ptr)?;
        lookup_executable_file(dfd, filename, flags)?
    };

    do_execve(elf_file, argv_ptr_ptr, envp_ptr_ptr, context)?;
    Ok(SyscallReturn::NoReturn)
}

fn lookup_executable_file(
    dfd: FileDesc,
    filename: String,
    flags: OpenFlags,
) -> Result<Arc<Dentry>> {
    let current = current!();
    let fs_resolver = current.fs().read();
    let dentry = if flags.contains(OpenFlags::AT_EMPTY_PATH) && filename.is_empty() {
        fs_resolver.lookup_from_fd(dfd)
    } else {
        let fs_path = FsPath::new(dfd, &filename)?;
        if flags.contains(OpenFlags::AT_SYMLINK_NOFOLLOW) {
            let dentry = fs_resolver.lookup_no_follow(&fs_path)?;
            if dentry.type_() == InodeType::SymLink {
                return_errno_with_message!(Errno::ELOOP, "the executable file is a symlink");
            }
            Ok(dentry)
        } else {
            fs_resolver.lookup(&fs_path)
        }
    }?;
    check_executable_file(&dentry)?;
    Ok(dentry)
}

fn do_execve(
    elf_file: Arc<Dentry>,
    argv_ptr_ptr: Vaddr,
    envp_ptr_ptr: Vaddr,
    context: &mut UserContext,
) -> Result<()> {
    let executable_path = elf_file.abs_path();
    let argv = read_cstring_vec(argv_ptr_ptr, MAX_ARGV_NUMBER, MAX_ARG_LEN)?;
    let envp = read_cstring_vec(envp_ptr_ptr, MAX_ENVP_NUMBER, MAX_ENV_LEN)?;
    debug!(
        "filename: {:?}, argv = {:?}, envp = {:?}",
        executable_path, argv, envp
    );
    // FIXME: should we set thread name in execve?
    let current_thread = current_thread!();
    let posix_thread = current_thread.as_posix_thread().unwrap();
    *posix_thread.thread_name().lock() =
        Some(ThreadName::new_from_executable_path(&executable_path)?);
    // clear ctid
    // FIXME: should we clear ctid when execve?
    *posix_thread.clear_child_tid().lock() = 0;

    let current = current!();

    // Ensure that the file descriptors with the close-on-exec flag are closed.
    let closed_files = current.file_table().lock().close_files_on_exec();
    for file in closed_files {
        file.clean_for_close()?;
    }

    debug!("load program to root vmar");
    let (new_executable_path, elf_load_info) = {
        let fs_resolver = &*current.fs().read();
        let process_vm = current.vm();
        load_program_to_vm(process_vm, elf_file.clone(), argv, envp, fs_resolver, 1)?
    };
    debug!("load elf in execve succeeds");

    let credentials = credentials_mut();
    set_uid_from_elf(&credentials, &elf_file)?;
    set_gid_from_elf(&credentials, &elf_file)?;

    // set executable path
    current.set_executable_path(new_executable_path);
    // set signal disposition to default
    current.sig_dispositions().lock().inherit();
    // set cpu context to default
    let default_content = UserContext::default();
    *context.general_regs_mut() = *default_content.general_regs();
    context.set_tls_pointer(default_content.tls_pointer());
    *context.fp_regs_mut() = *default_content.fp_regs();
    // set new entry point
    context.set_instruction_pointer(elf_load_info.entry_point() as _);
    debug!("entry_point: 0x{:x}", elf_load_info.entry_point());
    // set new user stack top
    context.set_stack_pointer(elf_load_info.user_stack_top() as _);
    debug!("user stack top: 0x{:x}", elf_load_info.user_stack_top());
    Ok(())
}

bitflags::bitflags! {
    struct OpenFlags: u32 {
        const AT_EMPTY_PATH = 0x1000;
        const AT_SYMLINK_NOFOLLOW = 0x100;
    }
}

fn read_filename(filename_ptr: Vaddr) -> Result<String> {
    let filename = read_cstring_from_user(filename_ptr, MAX_FILENAME_LEN)?;
    Ok(filename.into_string().unwrap())
}

fn read_cstring_vec(
    array_ptr: Vaddr,
    max_string_number: usize,
    max_string_len: usize,
) -> Result<Vec<CString>> {
    let mut res = Vec::new();
    let mut read_addr = array_ptr;
    let mut find_null = false;
    for _ in 0..max_string_number {
        let cstring_ptr = read_val_from_user::<usize>(read_addr)?;
        read_addr += 8;
        // read a null pointer
        if cstring_ptr == 0 {
            find_null = true;
            break;
        }
        let cstring = read_cstring_from_user(cstring_ptr, max_string_len)?;
        res.push(cstring);
    }
    if !find_null {
        return_errno_with_message!(Errno::E2BIG, "Cannot find null pointer in vector");
    }
    Ok(res)
}

/// Sets uid for credentials as the same of uid of elf file if elf file has `set_uid` bit.
fn set_uid_from_elf(credentials: &Credentials<WriteOp>, elf_file: &Arc<Dentry>) -> Result<()> {
    if elf_file.mode()?.has_set_uid() {
        let uid = elf_file.owner()?;
        credentials.set_euid(uid);
    }

    // No matter whether the elf_file has `set_uid` bit, suid should be reset.
    credentials.reset_suid();
    Ok(())
}

/// Sets gid for credentials as the same of gid of elf file if elf file has `set_gid` bit.
fn set_gid_from_elf(credentials: &Credentials<WriteOp>, elf_file: &Arc<Dentry>) -> Result<()> {
    if elf_file.mode()?.has_set_gid() {
        let gid = elf_file.group()?;
        credentials.set_egid(gid);
    }

    // No matter whether the the elf file has `set_gid` bit, sgid should be reset.
    credentials.reset_sgid();
    Ok(())
}
